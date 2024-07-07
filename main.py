from flask import Flask, render_template, request
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)

# Initialize the model and tokenizer for FLAN-T5
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# Function to generate questions for a given topic
def generate_questions_for_topic(topic, num_questions=29):
    input_text = f"generate questions on the topic: {topic}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=100, num_return_sequences=num_questions, num_beams=num_questions)
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return questions

@app.route('/', methods=['GET', 'POST'])
def index():
    questions = []
    if request.method == 'POST':
        topic = request.form['topic']
        questions = generate_questions_for_topic(topic)
    return render_template('index.html', questions=questions)


if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
