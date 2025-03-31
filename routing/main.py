from router import Router

def main():
    question = "What are the main factors influencing climate change?"
    router = Router(question)
    response = router.route()
    print(response)

if __name__ == "__main__":
    main()
    