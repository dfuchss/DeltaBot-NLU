import re
from threading import Lock, Thread
from typing import Dict, Any, Union

from flask import Flask, Blueprint, request, make_response
from flask_restx import Resource, Api, fields, abort
from rasa.model import get_model
from rasa.nlu.model import Interpreter

languages = ["de", "en"]

flask = Flask("MultiNLU")
api = Api(version="1.0", title="Multi NLU API", default_label="Fallback API", default="Fallback")

ns = api.namespace("nlu", "nlu namespace")

task = api.model("task", {
    "locale": fields.String(required=True, description=f"the locale; currently: {languages}"),
    "text": fields.String(required=True, description="the text to classify")
})


def load_thread(lock: Lock, model: str, key: str, ms: Dict[str, Union[Interpreter, Lock]]):
    lock.acquire()
    print(f"Loading model for Locale: {key}")
    interpreter = Interpreter.load(model)
    ms[key] = interpreter
    lock.release()


def load_models():
    ms = {}

    models = [f"{get_model('models_' + lang)}/nlu" for lang in languages]

    for idx, lang in enumerate(languages):
        lock = Lock()
        ms[lang] = lock

        thread = Thread(target=lambda: load_thread(lock, models[idx], lang, ms))
        thread.setDaemon(True)
        thread.start()

    return ms


nlu_models: Dict[str, Union[Interpreter, Lock]] = load_models()


def get_nlu(locale: str) -> Interpreter:
    lock_or_model = nlu_models[locale]

    if isinstance(lock_or_model, type(Lock())):
        lock_or_model.acquire()
        lock_or_model.release()

    return nlu_models[locale]


@ns.route("/")
class NLUEndpoint(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @api.expect(task)
    @api.response(200, "if recognition was successful")
    @api.response(404, "if Locale was not found")
    def post(self) -> Dict[str, Any]:
        locale = request.json["locale"]
        text = request.json["text"]

        if re.match(r"[a-z]{2}_[A-Z]{2}", locale) is not None:
            locale = locale[0:2]

        if locale not in languages:
            abort(404, f"Locale {locale} not found")
        result = get_nlu(locale).parse(text)
        # print(result)
        return result

    @api.response(200, f"Hello from MultiNLU {api.version}")
    @api.produces(['text/plain'])
    def get(self):
        response = make_response(f"Hello from MultiNLU {api.version}")
        response.headers.set("Content-Type", "text/plain")
        return response


if __name__ == '__main__':
    blueprint = Blueprint('api', __name__)

    api.init_app(blueprint)
    api.add_namespace(ns)

    flask.register_blueprint(blueprint)
    # Debug
    # flask.run(host="0.0.0.0", port=5005, debug=False)

    # Production
    from waitress import serve

    serve(flask, host="0.0.0.0", port=5005)
