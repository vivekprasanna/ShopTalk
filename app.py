from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import os
import mlProject.components.chat_bot as cb
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
api = Api(app, version="1.0", title="Shopping Assistant API", description="API for chatbot interactions")

# Swagger Namespace
ns = api.namespace("api", description="API operations")

# Request Model for Swagger
chatbot_model = api.model("ChatbotRequest", {
    "question": fields.String(required=True, description="User's question"),
    "chat_history": fields.List(fields.String, description="Previous chat messages")
})


@ns.route("/train")
class Training(Resource):
    def get(self):
        """Trigger model training pipeline."""
        os.system("python main.py")
        return {"message": "Training Successful!"}, 200


@ns.route("/submit")
class Chatbot(Resource):
    @api.expect(chatbot_model)
    def post(self):
        """Send a query to the chatbot."""
        try:
            data = request.json
            chat_history = data.get("chat_history", [])
            query = data.get("question")

            # Ensure response is clean
            answer, chat_history = cb.ask_question(query, chat_history)

            # Strip any problematic newlines
            clean_response = answer.replace("\n", " ").replace("\r", "")

            return jsonify({"answer": clean_response, "chat_history": chat_history})
        except Exception as e:
            return {"error": str(e)}, 500


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port = 8080, debug=True)
    app.run(host="0.0.0.0", port=8080)
