import nltk
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from data_access import fetch_GEO_ids, get_summaries, get_overall_design
import base64

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def custom_tokenizer(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

vectorizer = TfidfVectorizer(stop_words='english', tokenizer=custom_tokenizer, token_pattern=None)

def generate_clusters_and_visualization(fetch_design, n_clusters, pmid_list):
    print("Getting geo_ids")
    
    GEO_ids = fetch_GEO_ids(pmid_list)
    print("Getting geo_summaries")

    geo_summaries = get_summaries(GEO_ids)
    if fetch_design == 'yes':
        print("Getting overall_designs")
        for geo_summary in geo_summaries:
            geo_summary['overall_design'] = get_overall_design(geo_summary["accession"])
    else:
        for geo_summary in geo_summaries:
            geo_summary['overall_design'] = ""

    print("clustering")
    data = pd.DataFrame(geo_summaries)
    data = data.fillna('')
    data['combined_text'] = data['title'] + ' ' + data['organism'] + ' ' + data['summary'] + ' ' + data['overall_design']

    
    X = vectorizer.fit_transform(data['combined_text'])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X)
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X.toarray())
    
    cos_sim_matrix = cosine_similarity(X)

    title_dict = {i: data['title'][i] for i in range(len(data))}

    fig_3d = px.scatter_3d(
        data, 
        x=X_pca[:, 0], 
        y=X_pca[:, 1], 
        z=X_pca[:, 2], 
        color='Cluster',
        hover_data={'pubmedids': True, 'title': True}, 
        title="3D Clustering of Datasets Based on TF-IDF Vectors"
    )

    cos_sim_fig = go.Figure(data=go.Heatmap(
        z=cos_sim_matrix,
        x=list(range(len(data))),
        y=list(range(len(data))),
        colorscale='Viridis',
        colorbar=dict(title="Cosine Similarity"),
        text=cos_sim_matrix,
        hovertemplate="Cosine Similarity: %{z}<br>Row: %{y}<br>Col: %{x}<extra></extra>",
        showscale=True
    ))

    cos_sim_fig.update_layout(
        title="Cosine Similarity Matrix",
        xaxis_title="Dataset",
        yaxis_title="Dataset",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(data))),
            ticktext=[str(i) for i in range(len(data))]
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(data))),
            ticktext=[str(i) for i in range(len(data))]
        ),
        annotations=[
        go.layout.Annotation(
            x=x,
            y=y,
            text=str(round(cos_sim_matrix[y, x], 2)),
            showarrow=False,
            font=dict(size=10, color="white"),
            align="center"
        ) for y in range(len(data)) for x in range(len(data))
    ]
    )

    return fig_3d, cos_sim_fig, title_dict


def parse_pmid_file(contents):
    try:
        print(contents)
        content_string = contents.split(",")[1]

        decoded = base64.b64decode(content_string)
        decoded_str = decoded.decode("utf-8")
       
        return decoded_str.split('\n')
    
    except Exception as e:
        return []

app = Dash(__name__)

app.layout = html.Div([
    html.H1("3D Dataset Clustering Visualization"),

    dcc.Upload(
        id='upload-pmid-file',
        children=html.Button('Upload PMID File'),
        multiple=False
    ),

    html.Div([
        html.Label("Fetch overall_design:"),
        dcc.RadioItems(
            id='fetch-overall-design',
            options=[
                {'label': 'Yes', 'value': 'yes'},
                {'label': 'No', 'value': 'no'}
            ],
            value='no',
            labelStyle={'display': 'inline-block', 'marginRight': '10px'}
        )
    ], style={'marginTop': '20px'}),

    html.Div([
        html.Label("Number of Clusters:"),
        dcc.Input(
            id='num-clusters',
            type='number',
            min=1,
            step=1,
            value=3,
            style={'width': '100px'}
        )
    ], style={'marginTop': '20px'}),
    
    html.Button('Generate Visualization', id='submit_button', style={'marginTop': '10px'}),

    dcc.Loading(
        id="loading-spinner",
        type="circle",
        children=[
            html.Div([
                 dcc.Graph(
                    id='cluster_graph',
                    style={'height': '70vh', 'width': '100%'}
                ),
                html.Div([
                dcc.Graph(
                    id='cosine_similarity_graph',
                    style={'height': '70vh', 'width': '48%'}
                ), html.Div(
                id='titles_list',
                style={'width': '48%', 'marginLeft': '10px', 'marginTop': '20px'}
            )], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center'})
            ], style={'display': 'flex', 'flex-direction': 'column'}),
            
            
        ]
    ),
])

@app.callback(
    [Output('cluster_graph', 'figure'),
     Output('cosine_similarity_graph', 'figure'),
     Output('titles_list', 'children')],
    Input('submit_button', 'n_clicks'),
    State('fetch-overall-design', 'value'),
    State('num-clusters', 'value'),
    State('upload-pmid-file', 'contents'),
)


def update_graph(n_clicks, fetch_design, n_clusters, contents):
    if n_clicks is None:
        return {}, {}, ""

    if contents:
        pmid_list = parse_pmid_file(contents)
        if pmid_list:
            fig_3d, cos_sim_fig, title_dict = generate_clusters_and_visualization(fetch_design, n_clusters, pmid_list)
            
            title_list_html = html.Ul([html.Li(f"{i}: {title_dict[i]}") for i in range(len(title_dict))])

            return fig_3d, cos_sim_fig, title_list_html
    
    return {}, {}, ""


if __name__ == '__main__':
    app.run(debug=True)
