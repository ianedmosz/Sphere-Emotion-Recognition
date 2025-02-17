import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

app = Dash(__name__)  # Crear la instancia global de la aplicación Dash

class SphereGraph:
    def __init__(self, x, y, z, valence, dominance, arousal):
        self.x = x
        self.y = y
        self.z = z
        self.valence=valence
        self.dominance=dominance
        self.arousal=arousal

    def generate_sphere(self):
        fig = go.Figure()

        fig.add_trace(go.Surface(
            x=self.x,
            y=self.y,
            z=self.z,
            opacity=0.3,
            colorscale='Blues',
            showscale=False
        ))

        
        fig.add_trace(go.Scatter3d(
            x=[self.valence],
            y=[self.arousal],
            z=[self.dominance],
            mode='markers',
            marker=dict(
                size=6,
                color='red',
                opacity=1.0
            ),
            name='Emotion Point'))
        
        fig.update_layout(
            scene=dict(
                xaxis_title="Valence",
                yaxis_title="Dominance",
                zaxis_title="Arousal",
                camera=dict( #Posicion de la camara Inicial 
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )

            ))
        return fig

    def app_layout(self):
        app.layout = html.Div([ #Lay Out de la app
            dcc.Graph(
                id='sphere', 
                figure=self.generate_sphere(),
                style={'width': '100vw', 'height': '100vh'}

                ),
            dcc.Interval( #Actualizacion 
                id='interval-component',
                interval=1 * 1000,  # En milisegundos
                n_intervals=0
            )
        ])
        return app.layout

    def update_sphere(self):
        """Configurar la callback para actualizar la gráfica."""
        @app.callback(Output('sphere', 'figure'), [Input('interval-component', 'n_intervals')])
        def update_graph_live(n):
            print(f"Updating graph at interval: {n}")
            return self.generate_sphere()
