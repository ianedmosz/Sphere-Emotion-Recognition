import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import numpy as np
app = Dash(__name__)  # Crear la instancia global de la aplicación Dash

class SphereGraph:
    def __init__(self,valence, dominance, arousal):
        self.valence=valence
        self.dominance=dominance
        self.arousal=arousal
        self.x, self.y, self.z = self.generate_sphere_mesh()

    def generate_sphere_mesh(self):
        "Genera los puntos de la malla de la esfera"
        u=np.linespace(0,2*np.pi,30)
        v=np.linspace(0,np.pi,30)
        u,v=np.meshgrid(u,v)

        X=np.cos(u) * np.sin(v)
        Y=np.sin(u) * np.sin(v)
        Z=np.cos(v)

        return X,Y,Z
    
    def generate_circle(self,r=0.1,n_points=50):
        """Genera un círculo alrededor del punto de emoción en la esfera."""
        theta=np.linspace(0,2*np.pi,n_points)
        x_circle=self.valence+r*np.cos(theta)
        y_circle=self.arousal+r*np.sin(theta)
        z_circle=np.full_like(theta,self.dominance)

        return x_circle,y_circle,z_circle


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

        # Dibujar el punto de emoción
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
        
         # Dibujar el círculo alrededor del punto de emoción
        x_circle, y_circle, z_circle = self.generate_circle()
        fig.add_trace(go.Scatter3d(
            x=x_circle,
            y=y_circle,
            z=z_circle,
            mode='lines',
            line=dict(color='red', width=2),
            name='Emotion Region'
        ))

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
        """Define el layout de la aplicación Dash."""
        app.layout = html.Div([
            dcc.Graph(
                id='sphere', 
                figure=self.generate_sphere(),
                style={'width': '100vw', 'height': '100vh'}
            ),
            dcc.Interval(
                id='interval-component',
                interval=1 * 1000,  # Actualizar cada segundo
                n_intervals=0
            )
        ])
        return app.layout

    def update_sphere(self):
        """Configura la callback para actualizar la gráfica en tiempo real."""
        @app.callback(Output('sphere', 'figure'), [Input('interval-component', 'n_intervals')])
        def update_graph_live(n):
            print(f"Updating graph at interval: {n}")
            return self.generate_sphere()
