from flask import Flask, request, render_template, redirect, url_for, jsonify
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from werkzeug.utils import secure_filename
import os
import json
from flask_socketio import SocketIO, emit

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSeq2SeqLM,
    TrainerCallback
)
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import uuid

# new imports for sub process
import subprocess
import threading
import pusher
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = "./uploads"
pusher_client = pusher.Pusher(
    app_id='1672928',
    key='c6345026f4a44535826a',
    secret='22229a6ee51f6b80e6e1',
    cluster='us3',
    ssl=True
)

socketio = SocketIO(app, cors_allowed_origins="*")


@socketio.on('connect')
def handle_connect():
    print('connected')


@socketio.on('file_upload')
def handle_file_upload(data):
    file_name = data['file_name']
    file_content = data['file_content']
    #     store file in ../storage/app/datasets
    if os.name == "nt":
        file = "..\\storage\\app\\datasets\\" + file_name
    else:
        file = "../storage/app/datasets/" + file_name

    # if file exists then rename with random uuid
    if os.path.exists(file):
        file_name = str(uuid.uuid4()) +".csv"
        file = "../storage/app/datasets/" + file_name
        print("file exists")

    with open(file, 'w') as f:
        f.write(file_content)
    channel = data['channel'] + '_file_upload'

    message_to_emmit = {'file_path': file, 'file_name': data['file_name']}

    socketio.emit(channel, message_to_emmit)
    print(channel)
    print('file uploaded')

@socketio.on('file_download')
def handle_file_download(data):
    file_path = data['file_path']
#     read file
    with open(file_path, 'r') as f:
        file_content = f.read()
    channel = data['channel'] + '_file_download'

    message_to_emmit = {'file_content': file_content}

    socketio.emit(channel, message_to_emmit)
    print(channel)
    print('file downloaded')


@socketio.on('file_update')
def handle_file_update(data):
    file_path = data['file_path']
    file_content = data['file_content']
    #     store file in ../storage/app/datasets
    with open(file_path, 'w') as f:
        f.write(file_content)
    channel = data['channel'] + '_file_update'

    message_to_emmit = {'message': 'file updated'}

    socketio.emit(channel, message_to_emmit)

#     on any
@socketio.on('message')
def handle_message(data):
    print('received message: ' + data)


@socketio.on('event')
def handle_event(data):
    print('received event: ' + data)


def VerifyQueueAndRunning():
    #     verify if exists queue.json
    if os.path.exists('queue.json'):
        with open('queue.json', 'r') as f:
            queue = json.load(f)
    else:
        queue = []
        #     generate queue.json
        with open('queue.json', 'w') as f:
            json.dump(queue, f)

    if os.path.exists('running.json'):
        with open('running.json', 'r') as f:
            running = json.load(f)
            print("running", running)
    else:
        running = []
        #     generate running.json
        with open('running.json', 'w') as f:
            json.dump(running, f)
            print("running", running)
    return queue, running


def AddToQueue(params):
    #     get queue and running
    queue, running = VerifyQueueAndRunning()
    #     add new task to queue
    queue.append(params)
    #     save queue.json
    with open('queue.json', 'w') as f:
        json.dump(queue, f)
    return True


def RemoveFromQueue(random_filename):
    #     get queue and running
    queue, running = VerifyQueueAndRunning()
    #     remove task from queue
    for task in queue:
        if task['random_filename'] == random_filename:
            queue.remove(task)
    #     save queue.json
    with open('queue.json', 'w') as f:
        json.dump(queue, f)
    return True


def CreateAndAddToRunning(params):
    #     verify if exists running.json
    if os.path.exists('running.json'):
        with open('running.json', 'r') as f:
            running = json.load(f)
    else:
        running = []
    #     add new task to running
    running.append(params)
    #     save running.json
    with open('running.json', 'w') as f:
        json.dump(running, f)
    return True


def RemoveFromRunning(token):
    #     get running
    queue, running = VerifyQueueAndRunning()
    #     remove task from running

    for task in running:
        if task['model_token'] == token:
            running.remove(task)
    #     save running.json
    with open('running.json', 'w') as f:
        json.dump(running, f)
    return True


def GetFromQueue():
    #     verify if exists queue.json
    if os.path.exists('queue.json'):
        with open('queue.json', 'r') as f:
            queue = json.load(f)
    else:
        queue = []
    #     get first task from queue
    if len(queue) > 0:
        task = queue[0]
        del queue[0]
        #     save queue.json
        with open('queue.json', 'w') as f:
            json.dump(queue, f)
    else:
        task = {}
    return task


class CustomDataset(Dataset):
    def __init__(self, prompts, completions, tokenizer):
        self.prompts = prompts
        self.completions = completions
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        input_encoding = self.tokenizer(
            self.prompts[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        target_encoding = self.tokenizer(
            self.completions[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        return {
            "input_ids": input_encoding["input_ids"].flatten(),
            "attention_mask": input_encoding["attention_mask"].flatten(),
            "labels": target_encoding["input_ids"].flatten(),
        }


class PusherCallback(TrainerCallback):
    def __init__(self, args, model_token, user_id, api_url):
        self.args = args
        self.model_token = model_token
        self.user_id = user_id
        self.api_url = api_url

    def on_train_begin(self, args, state, control, **kwargs):
        data_status_request = {
            "status": "training begin",
        }
        req = requests.post(
            self.api_url + '/api/v1/models/' + self.model_token + '/status',
            data=data_status_request
        )

    def on_log(self, args, state, control, **kwargs):
        metrics = state.log_history
        metrics_string = json.dumps(metrics)

        pusher_message_json = {
            "epoch": state.epoch,
            "batch_size": args.per_device_train_batch_size,
            "learning_rate": args.learning_rate,
            "model_token": self.model_token,
            "status": "epoch end",
            "metrics": "",
        }

        #         if epoch it's divisible by 1
        if state.epoch % 1 == 0:
            pusher_channel = self.api_url + '_' + self.user_id
            #         remove http:// or https:// and . from url
            pusher_channel = pusher_channel.replace("https://", "")
            pusher_channel = pusher_channel.replace("http://", "")
            pusher_channel = pusher_channel.replace(".", "")

            pusher_client.trigger(pusher_channel, 'ai_model', {'message': pusher_message_json})
            data_status_request = {
                "status": "trained epoch " + str(state.epoch) + " of " + str(args.num_train_epochs),
                "metrics": metrics_string,
            }
            req = requests.post(
                self.api_url + '/api/v1/models/' + self.model_token + '/status',
                data=data_status_request
            )

    def on_train_end(self, args, state, control, **kwargs):
        data_status_request = {
            "status": "trained",
        }
        req = requests.post(
            self.api_url + '/api/v1/models/' + self.model_token + '/status',
            data=data_status_request
        )

        pusher_message_json = {
            "epoch": state.epoch,
            "batch_size": args.per_device_train_batch_size,
            "learning_rate": args.learning_rate,
            "model_token": self.model_token,
            "status": "training end",
            "metrics": "",
        }
        pusher_channel = self.api_url + '_' + self.user_id
        #         remove http:// or https:// and . from url
        pusher_channel = pusher_channel.replace("https://", "")
        pusher_channel = pusher_channel.replace("http://", "")
        pusher_channel = pusher_channel.replace(".", "")

        pusher_client.trigger(pusher_channel, 'ai_model', {'message': pusher_message_json})


def train(file, epoch, batch_size, learning_rate, random_filename, user_id, api_url):
    #     add to running
    CreateAndAddToRunning({
        "epoch": epoch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "file": file,
        "model_token": random_filename,
        "user_id": user_id,
        "api_url": api_url,
        "status": "running"
    })

    # Load the SQLcoder tokenizer and model
    train_base_checkpoint = "t5-small"
    device = "cuda"  # for GPU usage or "cpu" for CPU usage

    train_tokenizer = AutoTokenizer.from_pretrained(train_base_checkpoint)
    train_model = AutoModelForSeq2SeqLM.from_pretrained(train_base_checkpoint).to(device)

    # Read the training data
    data = pd.read_csv(file)

    # Split your data into training and evaluation sets
    train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)

    # Create the training and evaluation datasets
    train_dataset = CustomDataset(
        train_data["prompt"].tolist(),
        train_data["completion"].tolist(),
        train_tokenizer,
    )

    eval_dataset = CustomDataset(
        eval_data["prompt"].tolist(),
        eval_data["completion"].tolist(),
        train_tokenizer,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=epoch,  # Increasing to 100 based on size of dataset
        per_device_train_batch_size=batch_size,  # Experiment with this, depends on your GPU memory
        per_device_eval_batch_size=batch_size,  # Similarly this can be experimented with
        learning_rate=learning_rate,  # setting learning rate
        weight_decay=0.01,  # weight decay for regularization
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=2,
        gradient_accumulation_steps=1,  # Optional - can be used if experiencing memory issues
        remove_unused_columns=False,
        push_to_hub=False,
    )

    # Create a list of callbacks
    callbacks = [PusherCallback(args=training_args, model_token=random_filename, user_id=user_id, api_url=api_url)]

    # Create Trainer with both training and evaluation datasets
    trainer = Trainer(
        model=train_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Provide the evaluation dataset
        callbacks=callbacks,

    )

    # Fine-tune the model
    trainer.train()

    # Save the model
    path = "./models/" + random_filename  # hard-code output path
    model_path = path + "/model"
    tokenizer_path = path + "/tokenizer"

    train_model.save_pretrained(model_path)
    train_tokenizer.save_pretrained(tokenizer_path)

    #     remove from running
    RemoveFromRunning(random_filename)
    #     get next task from queue
    task = GetFromQueue()
    #     if exists task
    if len(task) > 0:
        #         create a thread to run the training
        thread = threading.Thread(
            target=train,
            args=(
                file,
                task['epoch'],
                task['batch_size'],
                task['learning_rate'],
                task['model_token'],
                task['user_id'],
                task['api_url'],
            ),
        )
        thread.start()
        return True
    else:
        return True


@app.route("/", methods=["GET", "POST"])
def index():
    res_dict = {"message": "Internal interface for training and serving the model"}
    return jsonify(res_dict)


@app.route("/api/v1/train", methods=["POST"])
def train_model():
    # if windows
    # file = "..\\..\\storage\\app\\public\\"+request.form.get("file")
    # if linux
    # file = "../../storage/app/public/"+request.form.get("file")


    file = request.form.get("file")
    # convert from path to file
    # file name its last part of path
    filename = file.split("/")[-1]

    if file:
        random_filename = str(uuid.uuid4())

        epoch = int(request.form.get("epoch"))
        batch_size = int(request.form.get("batch_size"))
        learning_rate = float(request.form.get("learning_rate"))
        user_id = request.form.get("user_id")

        api_url = request.form.get("api_url")

        # get queue and running
        queue, running = VerifyQueueAndRunning()
        print("queue", queue)

        #         verify running
        if len(running) >= 2:
            #         add to queue
            AddToQueue({
                "epoch": epoch,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "file": filename,
                "model_token": random_filename,
                "user_id": user_id,
                "api_url": api_url,
                "status": "added to queue"
            })
            res_dict = {
                "epoch": epoch,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "file": filename,
                "model_token": random_filename,
                "status": "success",
                "message": "added to queue"
            }
            return jsonify(res_dict)

        # new code for sub process
        # create a thread to run the training
        thread = threading.Thread(
            target=train,
            args=(
                file,
                epoch,
                batch_size,
                learning_rate,
                random_filename,
                user_id,
                api_url,
            ),
        )
        thread.start()

        res_dict = {
            "epoch": epoch,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "file": filename,
            "model_token": random_filename,
            "status": "success",
            "message": "training started"
        }
        return jsonify(res_dict)
    else:
        return jsonify({"status": "failed", "message": "no file uploaded"})

@app.route("/api/v1/models/<model_token>/status", methods=["POST"])
def model_status(model_token):
    # method options retuern cors headers
    if request.method == "OPTIONS":
        return jsonify({"status": "success"})
    # model path
    model_base_path = "./models/" + model_token
    model_path = model_base_path + "/model"
    tokenizer_path = model_base_path + "/tokenizer"
    # if model not found
    if not os.path.exists(model_path):
        res_dict = {
            "model_token": model_base_path,
            "status": "failed",
            "message": "model not found",
        }
        return jsonify(res_dict)
    status = request.form.get("status")
    metrics = request.form.get("metrics")
    # return json response
    res_dict = {"model_token": model_token, "status": status, "metrics": metrics}
    return jsonify(res_dict)



@app.route("/api/v1/prompt", methods=["GET", "POST"])
def prompt():
    # method options retuern cors headers
    if request.method == "OPTIONS":
        return jsonify({"status": "success"})

    # model path
    model_base_path = request.form.get("model_token")
    model_base_path = "./models/" + model_base_path
    model_path = model_base_path + "/model"
    tokenizer_path = model_base_path + "/tokenizer"

    max_length = int(request.form.get("max_tokens"))

    # temperature its a float value between 0 and 1 that represents the randomness of the generated text
    temperature = float(request.form.get("temperature"))
    #     fi temperature is 0.0 then set to 0.1
    if temperature == 0.0:
        temperature = 0.1

    # if model not found
    if not os.path.exists(model_path):
        res_dict = {
            "model_token": model_base_path,
            "status": "failed",
            "message": "model not found",
        }
        return jsonify(res_dict)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    prompt = request.form.get("prompt")
    output = model.generate(**tokenizer(prompt, return_tensors="pt"), max_length=max_length, temperature=temperature,
                            do_sample=True)
    completion = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # return json response
    res_dict = {"prompt": prompt, "completion": completion}
    return jsonify(res_dict)


@app.route("/api/v1/model/<model_token>", methods=["DELETE"])
def delete_model(model_token):
    model_token = "./models/" + model_token
    # if model not found
    if not os.path.exists(model_token):
        res_dict = {
            "model_token": model_token,
            "status": "failed",
            "message": "model not found",
        }
        return jsonify(res_dict)

    model_path = model_token + "/model"
    tokenizer_path = model_token + "/tokenizer"
    os.remove(model_path)
    os.remove(tokenizer_path)
    os.remove(model_token)
    res_dict = {"model_token": model_token, "status": "deleted"}
    return jsonify(res_dict)


if __name__ == "__main__":
    # app.run(debug=True,host='192.168.35.35', port=5000)
    socketio.run(app, host='0.0.0.0', port=8080)
