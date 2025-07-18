import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc



# Load data
df = pd.read_csv('dataweb.csv')

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
#app = dash.Dash(__name__)
app.title = "MetaScreen"


# Dropdown options
m_options = [{'label': m, 'value': m} for m in sorted(df['M'].unique())]
r1_options = [{'label': r1, 'value': r1} for r1 in sorted(df['R1'].unique())]
r2_options = [{'label': r2, 'value': r2} for r2 in sorted(df['R2'].unique())]
r3_options = [{'label': r3, 'value': r3} for r3 in sorted(df['R3'].unique())]



# Layout
app.layout = dbc.Container([
    html.H1("🔬 Structure–Barrier Interactive Analysis Platform for Metallocenes", className="text-center mb-20",
            style={'color': 'white', 'font-weight': 'bold'}),

    #
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Label("🔍 Search Catalyst Components",
                           style={'fontSize': 22,
                                  'fontWeight': 'bold',
                                  'color': '#ffffff',
                                  'marginBottom': '20px'}),
            ]),
            html.Label("Select Metal (M):",
                       style={'fontSize': 20,
                              'fontWeight': 'bold',
                              'color': '#ffffff',
                              'marginBottom': 4}),
            dcc.Dropdown(id='m-dropdown', options=m_options, multi=True),
            html.Br(),
            html.Label(["Select R", html.Sub("1"), ":"],
                       style={'fontSize': 20,
                              'fontWeight': 'bold',
                              'color': '#ffffff',
                              'marginBottom': 4}),
            dcc.Dropdown(id='r1-dropdown', options=r1_options, multi=True),
            html.Br(),
            html.Label(["Select R", html.Sub("2"), ":"],
                       style={'fontSize': 20,
                              'fontWeight': 'bold',
                              'color': '#ffffff',
                              'marginBottom': 4}),
            dcc.Dropdown(id='r2-dropdown', options=r2_options, multi=True),
            html.Br(),
            html.Label(["Select R", html.Sub("3"), ":"],
                       style={'fontSize': 20,
                              'fontWeight': 'bold',
                              'color': '#ffffff',
                              'marginBottom': 4}),
            dcc.Dropdown(id='r3-dropdown', options=r3_options, multi=True),
        ], width=4,
            style={
                'paddingRight': '20px',
                'height': '700px',
                'overflowY': 'auto',
                'background': 'linear-gradient(to top right, #000000, #2e003e)',
                'padding': '10px',
                'borderRadius': '10px'
            }),

        dbc.Col([
            dbc.CardHeader("📈 15 Lowest Barrier Combinations",
                           style={'fontSize': 22,
                                  'fontWeight': 'bold',
                                  'color': '#ffffff',
                                  'marginBottom': '20px'}),

            dcc.Graph(id='bar-chart', style={'height': '700px'})
        ], width=8)
    ]),

    dbc.Row([
        dbc.Col([
            html.Hr(style={'borderColor': 'gray'}),
            #html.H4("Barrier Across Catalyst Combinations (Scatter Plot)",style={'color': 'white', 'marginTop': '20px'}),
            html.Iframe(
                src="/assets/parallel_categories_barrier.html",
                style={"width": "100%", "height": "700px", "border": "none", "borderRadius": "10px"}
            )
        ], width=12)
    ]),

    # 新增底部满宽散点图
    dbc.Row([
        dbc.Col([
            html.Hr(style={'borderColor': 'gray'}),
            #html.H4("Barrier Across Catalyst Combinations (Scatter Plot)", style={'color': 'white', 'marginTop': '20px'}),
            html.Iframe(
                src="/assets/scatter_comboid_barrier.html",
                style={"width": "100%", "height": "700px", "border": "none", "borderRadius": "10px"}
            )
        ], width=12)
    ])

], fluid=True)


# Callback
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('m-dropdown', 'value'),
     Input('r1-dropdown', 'value'),
     Input('r2-dropdown', 'value'),
     Input('r3-dropdown', 'value')]
)
def update_bar(selected_m, selected_r1, selected_r2, selected_r3):
    filtered_df = df.copy()
    for col, sel in zip(['M', 'R1', 'R2', 'R3'], [selected_m, selected_r1, selected_r2, selected_r3]):
        if sel:
            filtered_df = filtered_df[filtered_df[col].isin(sel)]

    if not filtered_df.empty:
        lowest_15 = filtered_df.nsmallest(15, 'Barrier')[['M', 'R1', 'R2', 'R3', 'Barrier']].copy()
        lowest_15['label'] = lowest_15.apply(
            lambda x: f"{x['M']} | {x['R1']} | {x['R2']} | {x['R3']}", axis=1
        )
        fig = px.bar(
            lowest_15,
            x='Barrier',
            y='label',
            orientation='h',
            color='Barrier',
            color_continuous_scale='Viridis',
            labels={'Barrier': 'Barrier (kcal/mol)', 'label': 'Catalysts'}
        )
        fig.update_layout(
            font=dict(family='Arial', size=18, weight='bold', color='white'),
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='black',
            paper_bgcolor='black'
        )
    else:
        fig = px.bar(title="No matching data")
        fig.update_layout(
            font=dict(color='white'),
            plot_bgcolor='black',
            paper_bgcolor='black'
        )
    return fig


if __name__ == '__main__':
    app.run(debug=True, port=8050)
