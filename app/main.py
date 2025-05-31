
import dash
from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc

from utils.layout import layout

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = "FDIC Bank Health Explorer"

