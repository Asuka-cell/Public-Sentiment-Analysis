def main():
    import argparse
    import os
    import sys

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    from data_analysis.weibo.generate_enterprise_insights import generate as generate_weibo
    from data_analysis.zhihu.generate_enterprise_insights import generate as generate_zhihu

    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["weibo", "zhihu", "both"], default="both")
    parser.add_argument("--base_dir", default=os.getcwd())
    parser.add_argument("--out_dir", default=os.path.join(os.getcwd(), "dataset", "enterprise_insights"))
    parser.add_argument("--top_n_topics", type=int, default=30)
    parser.add_argument("--top_n_demands", type=int, default=20)
    args = parser.parse_args()

    platforms = ["weibo", "zhihu"] if args.platform == "both" else [args.platform]
    for p in platforms:
        if p == "weibo":
            t, d, tr = generate_weibo(args.out_dir, args.base_dir, args.top_n_topics, args.top_n_demands)
        else:
            t, d, tr = generate_zhihu(args.out_dir, args.base_dir, args.top_n_topics, args.top_n_demands)
        print(t)
        print(d)
        print(tr)


if __name__ == "__main__":
    main()
