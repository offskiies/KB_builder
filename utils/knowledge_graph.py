import os
from dotenv import load_dotenv
from typing import Optional
from tqdm import tqdm
import pandas as pd
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from langchain import OpenAI
from langchain.indexes.prompts.knowledge_triplet_extraction import (
    KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT,
)
from langchain.graphs.networkx_graph import KG_TRIPLE_DELIMITER


load_dotenv()


def check_if_kg_exists(texts):
    if not os.path.exists("data/knowledge_graph.csv"):
        openai = OpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=os.environ["OPENAI_API_KEY"],
            model_kwargs={"temperature": 1e-10},
        )

        triplets = []
        for passage in tqdm(texts):
            prompt = KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT.format_prompt(
                text=passage
            )
            result = openai(prompt.text)
            passage_triplets = [
                tuple(i.strip()[1:-1].split(", "))
                for i in result.split(KG_TRIPLE_DELIMITER)
                if len(i.strip()[1:-1].split(", ")) == 3
                and all([j != "none" for j in i.strip()[1:-1].split(", ")])
            ]
            triplets.extend(passage_triplets)

        kg_df = pd.DataFrame(
            set(triplets), columns=["source", "edge", "target"]
        )

        kg_df.to_csv("data/knowledge_graph.csv", index=False)


def plot_2D_kg(filter: Optional[str] = None):
    kg_df = pd.read_csv("data/knowledge_graph.csv")

    if not filter:
        filter_df = kg_df.copy()
    else:
        filter_df = kg_df.loc[
            (kg_df["source"].str.contains(filter, case=False))
            | (kg_df["edge"].str.contains(filter, case=False))
            | (kg_df["target"].str.contains(filter, case=False))
        ]

    G = nx.from_pandas_edgelist(
        filter_df, "source", "target", "edge", nx.MultiDiGraph(), "edge"
    )

    fig = plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, k=1.5)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=1500,
        edge_cmap=plt.cm.Blues,
        font_size=5,
    )
    edge_labels = {
        (i[0], i[1]): i[2]
        if len(i[2].split()) <= 6
        else " ".join(i[2].split()[:6]) + "..."
        for i in list(G.edges)
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels, font_color="red", font_size=5
    )

    return fig


def plot_3D_kg(filter: Optional[str] = None):
    kg_df = pd.read_csv("data/knowledge_graph.csv")

    if not filter:
        filter_df = kg_df.copy()
    else:
        filter_df = kg_df.loc[
            (kg_df["source"].str.contains(filter, case=False))
            | (kg_df["edge"].str.contains(filter, case=False))
            | (kg_df["target"].str.contains(filter, case=False))
        ]

    G = nx.from_pandas_edgelist(
        filter_df, "source", "target", "edge", nx.MultiDiGraph(), "edge"
    )

    spring_3D = nx.spring_layout(G, dim=3, k=0.5)

    x_nodes = [
        spring_3D[key][0] for key in spring_3D.keys()
    ]  # x-coordinates of nodes
    y_nodes = [spring_3D[key][1] for key in spring_3D.keys()]  # y-coordinates
    z_nodes = [spring_3D[key][2] for key in spring_3D.keys()]  # z-coordinates

    trace_nodes = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode="markers",
        marker=dict(symbol="circle", size=12, color="skyblue"),
        text=list(spring_3D.keys()),
        hoverinfo="text",
    )

    fig = go.Figure(data=trace_nodes)

    xtp, ytp, ztp = [], [], []
    spacings = 1000

    for edge in G.edges():
        xtp.append(
            [
                i * spring_3D[edge[0]][0] + (1 - i) * spring_3D[edge[1]][0]
                for i in np.linspace(0, 1, spacings)
            ]
        )

        ytp.append(
            [
                i * spring_3D[edge[0]][1] + (1 - i) * spring_3D[edge[1]][1]
                for i in np.linspace(0, 1, spacings)
            ]
        )

        ztp.append(
            [
                i * spring_3D[edge[0]][2] + (1 - i) * spring_3D[edge[1]][2]
                for i in np.linspace(0, 1, spacings)
            ]
        )

    etext = [i[2] for i in list(G.edges)]

    xtp_matrix = np.array(xtp)
    ytp_matrix = np.array(ytp)
    ztp_matrix = np.array(ztp)

    for i in range(spacings):
        colour_scaler = i / spacings
        trace_weights = go.Scatter3d(
            x=xtp_matrix[:, i],
            y=ytp_matrix[:, i],
            z=ztp_matrix[:, i],
            mode="markers",
            marker=dict(
                color=f"rgba({255*(1-colour_scaler)}, {255*colour_scaler}, 0, 0.9)",
                size=1,
            ),
            text=etext,
            hoverinfo="text",
        )

        fig.add_trace(trace_weights)

    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
    )

    return fig
