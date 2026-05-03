import argparse
import os

from zhihu_common import get_zhihu_cookies_interactive


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cookies_path = os.path.join(base_dir, "zhihu_cookies.json")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cookies", default=cookies_path)
    parser.add_argument("--use_profile", action="store_true")
    parser.add_argument("--user_data_dir", default=None)
    parser.add_argument("--profile_dir", default=None)
    parser.add_argument("--non_interactive", action="store_true")
    parser.add_argument("--wait_seconds", type=int, default=600)
    args = parser.parse_args()

    if args.use_profile:
        if args.user_data_dir:
            os.environ["CHROME_USER_DATA_DIR"] = str(args.user_data_dir)
        else:
            os.environ.setdefault(
                "CHROME_USER_DATA_DIR",
                os.path.expanduser("~/Library/Application Support/Google/Chrome"),
            )
        if args.profile_dir:
            os.environ["CHROME_PROFILE_DIR"] = str(args.profile_dir)
        else:
            os.environ.setdefault("CHROME_PROFILE_DIR", "Default")
    get_zhihu_cookies_interactive(
        cookies_path,
        non_interactive=bool(args.non_interactive),
        wait_seconds=int(args.wait_seconds),
    )
    if str(args.cookies) != str(cookies_path):
        get_zhihu_cookies_interactive(
            str(args.cookies),
            non_interactive=bool(args.non_interactive),
            wait_seconds=int(args.wait_seconds),
        )

if __name__ == "__main__":
    main()
