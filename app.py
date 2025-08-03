from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    print("Received webhook:", data)
    return jsonify({'status': 'ok'}), 200

@app.route('/')
def home():
    return 'Webhook Bot is running!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
