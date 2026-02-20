import os

from dotenv import load_dotenv

load_dotenv()


def main():
    print("Hello from langchain-course!")
    print(os.environ.get("GOOGLE_GENAI_API_KEY"))


if __name__ == "__main__":
    main()
