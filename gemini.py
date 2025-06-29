from google import genai
import dotenv

dotenv.load_dotenv()

def geminiCall(content: str) -> str:
    """
    Calls the Gemini API to generate content based on the provided input.

    :param content: The input content to be processed by the Gemini model.
    :return: The generated content as a string.
    """
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents="Explain how AI works in a few words"
    )
    print(response.text)

if __name__ == "__main__":
    geminiCall("Are you working respond with chicken if you are")