from flask import Flask, render_template, request, Response
import json
import numpy as np
from srn import SRN
from sklearn.cluster import KMeans
import plotly.express as px
import json
import plotly.io as pio



def create_plotly_figure(states_2d, clusters):
    fig = px.scatter(
        x=states_2d[:, 0].tolist(),
        y=states_2d[:, 1].tolist(),  
        color=clusters.astype(str),
        title="Hidden State Clustering"
    )
    return fig.to_plotly_json() 

app = Flask(__name__)

srn_model = None
token_to_idx = {}
idx_to_token = {}
vocab_size = 0

def cluster_states(states, n_clusters=5):

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(states)

    return clusters


def one_hot(token, token_to_idx, vocab_size):
    vec = np.zeros(vocab_size)
    vec[token_to_idx[token]] = 1
    return vec

def collect_hidden_states(srn_model, sequences, token_to_idx, vocab_size):

    states = []
    labels = []

    for seq in sequences:

        srn_model.reset_context()

        for token in seq:

            x = one_hot(token, token_to_idx, vocab_size)

            h, _ = srn_model.forward(x)

            states.append(h.flatten())
            labels.append(token)

            srn_model.context = h

    return np.array(states), labels

from sklearn.decomposition import PCA

def reduce_dim(states):

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(states)

    return reduced



@app.route("/", methods=["GET","POST"])
def index():

    global srn_model, token_to_idx, idx_to_token, vocab_size

    result = ""

    if request.method == "POST":


        if "predict" in request.form and srn_model:

            token = request.form["token"]
        
            try:
                # One-hot encode the input token
                x = one_hot(token, token_to_idx, vocab_size)
        
                # Forward pass
                h, y = srn_model.forward(x)
        
                # Update context
                srn_model.context = h
        
                # Get predicted token
                pred = idx_to_token[np.argmax(y)]
        
                result = f"Prediction: {pred}"
                print(result)
        
            except KeyError:
                # Handle unknown token
                result = f"Unknown token: '{token}'. Please use a token from the training set."
                print(result)
    
    return render_template("index.html", result=result)

@app.route("/train_stream", methods=["POST"])
def train_stream():

    global srn_model, token_to_idx, idx_to_token, vocab_size

    file = request.files["dataset"]
    data = json.load(file)

    sequences = data["extracted_sequences"]

    hidden_size = int(request.form["hidden_size"])
    lr = float(request.form["lr"])
    mu = float(request.form["mu"])
    clearval = float(request.form["clearval"])
    epochs = int(request.form["epochs"])

    tokens = sorted(set(t for seq in sequences for t in seq))
    token_to_idx = {t:i for i,t in enumerate(tokens)}
    vocab_size = len(tokens)
    print(vocab_size)
    idx_to_token = {i:t for t,i in token_to_idx.items()}

    srn_model = SRN(vocab_size, hidden_size, lr, mu, clearval)

    def generate():

        for epoch in range(epochs):

            total_loss = 0

            for seq in sequences:

                srn_model.reset_context()

                for i in range(len(seq)-1):

                    x = one_hot(seq[i], token_to_idx, vocab_size)
                    target = one_hot(seq[i+1], token_to_idx, vocab_size)

                    total_loss += srn_model.train_step(x, target)

            progress = int((epoch+1)/epochs*100)

            yield f"data: {json.dumps({'epoch':epoch+1,'loss':total_loss,'progress':progress})}\n\n"


        states, labels = collect_hidden_states(srn_model, sequences, token_to_idx, vocab_size)
        clusters = cluster_states(states, n_clusters=6)
        states_2d = reduce_dim(states)

        
        plot_data = create_plotly_figure(states_2d, clusters)
        
        yield f"data: {json.dumps({'plot': plot_data})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"


    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True)