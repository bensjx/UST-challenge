# Import general libraries
import pymysql
import pandas as pd
import numpy as np
import copy
import ast
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import dash
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])

app.title = "UFlix"
app_name = "UFlix"
server = app.server
LOGO = "./assets/logo.png"

""" Start of config """

""" End of config """


""" Start of helper functions """


def get_reco(title, cs_matrix, indices_dict, df):
    """
    Given a movie title and a pre-trained similarity matrix, return the top 10 most simililar movies
    Input:
        1. title in string
        2. cosine similarity matrix (Fitted)
        3. dictionary of mapping of title to index
        4. dataframe to retrieve title given index
    Output:
        1. list of top 10 movie titles
    """
    if title == None:
        return ["Generating results..."]
    # Get the index of the movie that matches the title
    idx = indices_dict[title]

    # Get the similarity scores of all 10K movies that are related to this movie & sort it & return top 10
    sim_scores = sorted(
        list(enumerate(cs_matrix[idx])), key=lambda x: x[1], reverse=True
    )[1:11]

    # top 10 movie indices
    movie_indices = [i[0] for i in sim_scores]

    # top 10 movie titles
    return df["title"].iloc[movie_indices]


""" End of helper functions """

# Navigation bar
navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=LOGO, height="50px")),
                    dbc.Col(
                        dbc.NavbarBrand(
                            "UFlix",
                            className="ml-auto",
                            style={"font-size": 30},
                        )
                    ),
                ],
                align="left",
                no_gutters=True,
            ),
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
    ],
    color="dark",
    dark=True,
    style={"width": "100%"},
)

recoTab = html.Div(
    children=[
        html.H2("Recommendation System", style={"textAlign": "center"}),
        # List of random movies with refresh button
        html.H3(
            "List of random movie names: Click the refresh button to generate new list"
        ),
        html.Div(id="rec-list"),
        html.Button(
            "Refresh", id="button_refresh", className="btn btn-success btn-lg btn-block"
        ),
        html.Br(),
        html.Div(
            "Key in a movie title and you will be returned with the top 10 most related movie."
        ),
        dbc.Textarea(
            id="movie-input",
            className="mb-3",
            placeholder="Input in lower caps e.g. barfly",
        ),
        html.Button(
            "Generate",
            id="button_generate",
            className="btn btn-success btn-lg btn-block",
        ),
        html.Br(),
        html.Div(id="rec-table"),
    ],
    style={"padding": "20px"},
)


##### For recoTab recommender
@app.callback(
    Output("rec-table", "children"),
    [
        Input("button_generate", "n_clicks"),
        Input("movie-input", "value"),
    ],  # upon clicking search button
)
def rec_table(n, input_val):
    # read data from cloud sql
    con = pymysql.connect(host="34.87.57.222", user="bensjyy", passwd="", db="movies")
    query = "SELECT * FROM reco"
    rec_df_small = pd.read_sql(query, con)

    # build vectorizer
    cv = CountVectorizer(
        stop_words="english"
    )  # set up a count vectorizer and remove stop words
    cv_corpus = cv.fit_transform(rec_df_small["corpus"])
    cossim_matrix = cosine_similarity(cv_corpus, cv_corpus)

    # Create a mapping of movie title to index
    indices = pd.Series(rec_df_small.index, index=rec_df_small.title.values).to_dict()

    res = get_reco(input_val, cossim_matrix, indices, rec_df_small)

    children = html.Div(
        [
            html.H4("The top 10 movies are:"),
            html.Ol(id="reco-list", children=[html.Li(i) for i in res]),
        ]
    )
    con.close()
    return children


##### For recoTab random list of words
@app.callback(Output("rec-list", "children"), [Input("button_refresh", "n_clicks")])
def rec_list(n):
    # read data from cloud sql
    con = pymysql.connect(host="34.87.57.222", user="bensjyy", passwd="", db="movies")
    query = "SELECT * FROM reco"
    rec_df_small = pd.read_sql(query, con)
    children = html.Div(
        html.Ul(
            id="sample-list",
            children=[html.Li(i) for i in rec_df_small.title.sample(10)],
        )
    )
    con.close()
    return children


# Layout of entire app
app.layout = html.Div(
    [
        navbar,
        dbc.Tabs(
            [
                dbc.Tab(recoTab, id="label_tab1", label="Recommendation System"),
                # dbc.Tab(predTab, id="label_tab2", label="Prediction"),
            ],
            style={"font-size": 20, "background-color": "#b9d9eb"},
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
