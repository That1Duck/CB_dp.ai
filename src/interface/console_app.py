#from src.interface.run_query import run_query
from src.interface.run_query import run_query

def run():
    print("Enter your query or path to image. For exit use Ctrl + C:")
    while True:
        user_input = input("> ").strip()
        res = run_query(user_input)

        if res.get("status") == "ok":
            print("\nRecomendation:")
            print(f"Title: {res.get('title', '')}")
            print(f"Why:   {res.get('why', '')}")
            print(f"URL:   {res.get('url', '')}")
            print("\n(Top-k candidates)")
            for i, c in enumerate(res.get("candidates", []), 1):
                print(f"[{i}] {c['title']} — {c['url']}")

        else:
            print(f"Error: {res.get('message')}\n")
