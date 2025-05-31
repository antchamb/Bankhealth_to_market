
from dash import html, dcc




parameters = html.Div([
    html.H3('Adjust Params:'),
    html.Div([
        html.Label('bank name:'),
        dcc.Input(
            id='bank-name',
            type='text',
            value='',
            style={'marginBottom': '10px', 'width': '100%'}
        )
    ])
])