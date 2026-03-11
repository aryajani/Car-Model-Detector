from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_car_info(car_name):

    prompt = f"""
    Give a short explanation of the car {car_name}.
    Include:
    - manufacturer
    - engine options
    - year range
    - notable features
    
    Keep it under 180 words.
    """

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return completion.choices[0].message.content