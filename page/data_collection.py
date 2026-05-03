import os
import subprocess
import time
import select
from datetime import datetime
from typing import Dict, Optional, Tuple

import streamlit as st


def _proc_key(name: str) -> str:
    return f"__dc_proc__{name}"


def _log_key(name: str) -> str:
    return f"__dc_log__{name}"


def _meta_key(name: str) -> str:
    return f"__dc_meta__{name}"


def _get_proc(name: str):
    return st.session_state.get(_proc_key(name))


def _set_proc(name: str, p):
    st.session_state[_proc_key(name)] = p


def _append_log(name: str, text: str) -> None:
    k = _log_key(name)
    st.session_state[k] = (st.session_state.get(k) or "") + (text or "")


def _get_log(name: str) -> str:
    return str(st.session_state.get(_log_key(name)) or "")


def _set_meta(name: str, meta: Dict) -> None:
    st.session_state[_meta_key(name)] = meta


def _get_meta(name: str) -> Dict:
    v = st.session_state.get(_meta_key(name))
    return v if isinstance(v, dict) else {}


def _start_cmd(name: str, cmd, cwd: str, input_text: Optional[str] = None) -> None:
    cmd = list(cmd)
    exe0 = str(cmd[0]) if cmd else ""
    exe0_base = os.path.basename(exe0)
    is_python = exe0_base in {"python", "python3"} or exe0_base.startswith("python")
    if cmd and is_python and "-u" not in cmd:
        cmd = [cmd[0], "-u"] + cmd[1:]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )
    if input_text is not None and p.stdin is not None:
        try:
            p.stdin.write(str(input_text))
            p.stdin.flush()
            p.stdin.close()
        except Exception:
            pass
    _set_proc(name, p)
    _set_meta(name, {"started_at": time.time(), "cmd": cmd, "cwd": cwd})


def _poll_cmd(name: str) -> None:
    p = _get_proc(name)
    if p is None:
        return
    if p.stdout is None:
        return
    try:
        while True:
            r, _, _ = select.select([p.stdout], [], [], 0)
            if not r:
                break
            line = p.stdout.readline()
            if not line:
                break
            _append_log(name, line)
    except Exception:
        pass
    rc = p.poll()
    if rc is not None:
        try:
            if p.stdout is not None:
                rest = p.stdout.read() or ""
                if rest:
                    _append_log(name, rest)
        except Exception:
            pass


def _stop_cmd(name: str) -> None:
    p = _get_proc(name)
    if p is None:
        return
    try:
        p.terminate()
    except Exception:
        pass


def _file_info(path: str) -> str:
    if not os.path.exists(path):
        return "不存在"
    try:
        st_size = os.path.getsize(path)
        mtime = os.path.getmtime(path)
        dt = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        return f"存在 | {st_size/1024:.1f} KB | 修改时间 {dt}"
    except Exception:
        return "存在"


def render_data_collection(base_dir: str) -> None:
    st.title("🕸️ 数据采集")

    project_dir = base_dir
    weibo_dir = os.path.join(project_dir, "get_weibo_data")
    zhihu_dir = os.path.join(project_dir, "get_zhihu_data")
    dataset_dir = os.path.join(project_dir, "dataset")

    tab_weibo, tab_zhihu = st.tabs(["微博", "知乎"])

    with tab_weibo:
        st.subheader("微博采集")

        cookies_path = os.path.join(weibo_dir, "cookies.txt")
        st.write(f"cookies.txt: {_file_info(cookies_path)}")
        st.warning("采集前请先扫码/登录微博并生成 cookies.txt。")

        login_wait = st.number_input("登录等待时长（秒）", min_value=30, max_value=1800, value=600, step=30)
        auto_refresh = st.checkbox("自动刷新日志", value=True, key="__dc_weibo_auto_refresh")

        login_name = "weibo_login"
        if st.button("0) 打开浏览器登录微博并导出 cookies", use_container_width=True):
            st.session_state[_log_key(login_name)] = ""
            _start_cmd(
                login_name,
                ["python3", os.path.join(weibo_dir, "GetWeiboCookies.py"), "--non_interactive", "--wait_seconds", str(int(login_wait))],
                cwd=weibo_dir,
            )

        p_login = _get_proc(login_name)
        if p_login is not None:
            _poll_cmd(login_name)
            rc = p_login.poll()
            st.text_area("登录日志", value=_get_log(login_name), height=220)
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("暂停/停止登录进程", use_container_width=True, key="__dc_stop_weibo_login"):
                    _stop_cmd(login_name)
                    st.rerun()
            with col_b:
                if st.button("刷新登录日志", use_container_width=True, key="__dc_refresh_weibo_login"):
                    st.rerun()
            if rc is None and auto_refresh:
                time.sleep(1.0)
                st.rerun()

        keyword = st.text_input("搜索关键词", value="西贝预制菜")
        max_pages = st.number_input("最大页数（0 表示不限制）", min_value=0, max_value=500, value=0, step=1)

        ids_local = os.path.join(weibo_dir, "weibo_ids.txt")
        st.write(f"weibo_ids.txt（采集目录）: {_file_info(ids_local)}")

        idlist_name = "weibo_idlist"
        if st.button("1) 搜索并生成 weibo_ids.txt", use_container_width=True):
            if not os.path.exists(cookies_path):
                st.error("未找到 get_weibo_data/cookies.txt，请先执行步骤 0 登录微博。")
            elif not keyword.strip():
                st.error("关键词不能为空")
            else:
                st.session_state[_log_key(idlist_name)] = ""
                _start_cmd(
                    idlist_name,
                    ["python3", os.path.join(weibo_dir, "GetWeiboIdList.py")],
                    cwd=weibo_dir,
                    input_text=str(keyword).strip() + "\n",
                )

        p_id = _get_proc(idlist_name)
        if p_id is not None:
            _poll_cmd(idlist_name)
            rc = p_id.poll()
            st.text_area("ID 搜索日志", value=_get_log(idlist_name), height=260)
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("暂停/停止搜索进程", use_container_width=True, key="__dc_stop_weibo_idlist"):
                    _stop_cmd(idlist_name)
                    st.rerun()
            with col_b:
                if st.button("刷新搜索日志", use_container_width=True, key="__dc_refresh_weibo_idlist"):
                    st.rerun()
            if rc is None and auto_refresh:
                time.sleep(1.0)
                st.rerun()

        st.divider()

        st.write(f"dataset/weibo_posts.csv: {_file_info(os.path.join(dataset_dir, 'weibo_posts.csv'))}")
        st.write(f"dataset/weibo_comments.csv: {_file_info(os.path.join(dataset_dir, 'weibo_comments.csv'))}")

        crawl_name = "weibo_crawl"
        if st.button("2) 抓取微博帖子与评论", use_container_width=True):
            if not os.path.exists(cookies_path):
                st.error("未找到 get_weibo_data/cookies.txt，请先执行步骤 0 登录微博。")
            else:
                if not os.path.exists(ids_local):
                    st.error("未找到 weibo_ids.txt（get_weibo_data/weibo_ids.txt），请先执行步骤 1。")
                else:
                    st.session_state[_log_key(crawl_name)] = ""
                    _start_cmd(crawl_name, ["python3", os.path.join(weibo_dir, "GetWeiboDataset.py")], cwd=weibo_dir)

        p_crawl = _get_proc(crawl_name)
        if p_crawl is not None:
            _poll_cmd(crawl_name)
            rc = p_crawl.poll()
            st.text_area("抓取日志", value=_get_log(crawl_name), height=320)
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("暂停/停止抓取进程", use_container_width=True, key="__dc_stop_weibo_crawl"):
                    _stop_cmd(crawl_name)
                    st.rerun()
            with col_b:
                if st.button("刷新抓取日志", use_container_width=True, key="__dc_refresh_weibo_crawl"):
                    st.rerun()
            if rc is None and auto_refresh:
                time.sleep(1.0)
                st.rerun()

        st.caption("输出位置：dataset/weibo_posts.csv 与 dataset/weibo_comments.csv")

    with tab_zhihu:
        st.subheader("知乎采集")

        cookies_path = os.path.join(zhihu_dir, "zhihu_cookies.json")
        targets_path = os.path.join(zhihu_dir, "zhihu_targets.txt")
        st.write(f"zhihu_cookies.json: {_file_info(cookies_path)}")
        st.write(f"zhihu_targets.txt: {_file_info(targets_path)}")
        st.write(f"dataset 目录: {_file_info(dataset_dir)}")
        st.warning("采集前请先扫码/登录知乎并生成 zhihu_cookies.json。")

        zh_login_wait = st.number_input("知乎登录等待时长（秒）", min_value=30, max_value=1800, value=600, step=30)
        zh_auto_refresh = st.checkbox("自动刷新日志", value=True, key="__dc_zhihu_auto_refresh")

        zh_login_name = "zhihu_login"
        if st.button("0) 打开浏览器登录知乎并导出 cookies", use_container_width=True):
            st.session_state[_log_key(zh_login_name)] = ""
            _start_cmd(
                zh_login_name,
                ["python3", os.path.join(zhihu_dir, "GetZhihuCookies.py"), "--non_interactive", "--wait_seconds", str(int(zh_login_wait))],
                cwd=zhihu_dir,
            )

        p_zh_login = _get_proc(zh_login_name)
        if p_zh_login is not None:
            _poll_cmd(zh_login_name)
            rc = p_zh_login.poll()
            st.text_area("登录日志", value=_get_log(zh_login_name), height=220)
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("暂停/停止登录进程", use_container_width=True, key="__dc_stop_zhihu_login"):
                    _stop_cmd(zh_login_name)
                    st.rerun()
            with col_b:
                if st.button("刷新登录日志", use_container_width=True, key="__dc_refresh_zhihu_login"):
                    st.rerun()
            if rc is None and zh_auto_refresh:
                time.sleep(1.0)
                st.rerun()

        kw = st.text_input("知乎搜索关键词", value="西贝 预制菜")
        search_limit = st.number_input("每页/每次搜索上限", min_value=1, max_value=50, value=20, step=1)
        search_pages = st.number_input("搜索页数", min_value=1, max_value=20, value=3, step=1)
        search_types = st.text_input("搜索类型（逗号分隔）", value="question,answer")

        zh_id_name = "zhihu_idlist"
        if st.button("1) 搜索并写入 zhihu_targets.txt", use_container_width=True):
            if not os.path.exists(cookies_path):
                st.error("未找到 get_zhihu_data/zhihu_cookies.json，请先执行步骤 0 登录知乎。")
            elif not kw.strip():
                st.error("关键词不能为空")
            else:
                st.session_state[_log_key(zh_id_name)] = ""
                cmd = [
                    "python3",
                    os.path.join(zhihu_dir, "GetZhihuIdList.py"),
                    "--cookies",
                    cookies_path,
                    "--targets",
                    targets_path,
                    "--keyword",
                    kw.strip(),
                    "--search_limit",
                    str(int(search_limit)),
                    "--search_pages",
                    str(int(search_pages)),
                    "--search_types",
                    str(search_types).strip(),
                ]
                _start_cmd(zh_id_name, cmd, cwd=zhihu_dir)

        p_zh_id = _get_proc(zh_id_name)
        if p_zh_id is not None:
            _poll_cmd(zh_id_name)
            rc = p_zh_id.poll()
            st.text_area("搜索日志", value=_get_log(zh_id_name), height=260)
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("暂停/停止搜索进程", use_container_width=True, key="__dc_stop_zhihu_idlist"):
                    _stop_cmd(zh_id_name)
                    st.rerun()
            with col_b:
                if st.button("刷新搜索日志", use_container_width=True, key="__dc_refresh_zhihu_idlist"):
                    st.rerun()
            if rc is None and zh_auto_refresh:
                time.sleep(1.0)
                st.rerun()

        st.divider()

        top_answers = st.number_input("每个问题抓取 TopK 高赞回答", min_value=1, max_value=50, value=10, step=1)
        answers_pages = st.number_input("回答翻页最大页数", min_value=1, max_value=50, value=5, step=1)
        prefer_selenium = st.checkbox("优先使用 Selenium（更稳但更慢）", value=False)
        refetch_missing = st.checkbox("补抓缺失字段（title/publish_time 等）", value=True)

        st.write(f"zhihu_questions.csv: {_file_info(os.path.join(dataset_dir, 'zhihu_questions.csv'))}")
        st.write(f"zhihu_answers.csv: {_file_info(os.path.join(dataset_dir, 'zhihu_answers.csv'))}")

        zh_crawl_name = "zhihu_crawl"
        if st.button("2) 抓取知乎问题与回答数据", use_container_width=True):
            if not os.path.exists(cookies_path):
                st.error("未找到 get_zhihu_data/zhihu_cookies.json，请先执行步骤 0 登录知乎。")
            elif not os.path.exists(targets_path):
                st.error("未找到 zhihu_targets.txt，请先执行步骤 1 搜索目标。")
            else:
                st.session_state[_log_key(zh_crawl_name)] = ""
                cmd = [
                    "python3",
                    os.path.join(zhihu_dir, "GetZhihuDataset.py"),
                    "--targets",
                    targets_path,
                    "--cookies",
                    cookies_path,
                    "--output_dir",
                    dataset_dir,
                    "--answers_pages",
                    str(int(answers_pages)),
                    "--top_answers",
                    str(int(top_answers)),
                ]
                if prefer_selenium:
                    cmd.append("--prefer_selenium")
                if refetch_missing:
                    cmd.append("--refetch_missing")
                _start_cmd(zh_crawl_name, cmd, cwd=zhihu_dir)

        p_zh_crawl = _get_proc(zh_crawl_name)
        if p_zh_crawl is not None:
            _poll_cmd(zh_crawl_name)
            rc = p_zh_crawl.poll()
            st.text_area("抓取日志", value=_get_log(zh_crawl_name), height=320)
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("暂停/停止抓取进程", use_container_width=True, key="__dc_stop_zhihu_crawl"):
                    _stop_cmd(zh_crawl_name)
                    st.rerun()
            with col_b:
                if st.button("刷新抓取日志", use_container_width=True, key="__dc_refresh_zhihu_crawl"):
                    st.rerun()
            if rc is None and zh_auto_refresh:
                time.sleep(1.0)
                st.rerun()

        st.caption("输出位置：dataset/zhihu_questions.csv 与 dataset/zhihu_answers.csv")
