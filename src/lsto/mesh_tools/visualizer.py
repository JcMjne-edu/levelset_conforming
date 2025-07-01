import plotly.graph_objects as go

def plot_tetramesh_intersection(connect,coord,y_intersection,angle=None):
  """
  connect : (m,4)
  coord : (n,3)
  y_intersection : (m,3)
  angle : (m,)
  """
  face=connect[:,[0,1,2,0,3,1,0,2,3,1,3,2]].reshape(-1,3) # (m*4,3)
  coord_face=coord[face] # (m*4,3,3)
  msk_face_surface=(coord_face[:,:,1]>y_intersection).all(axis=1) # (m*4,)
  msk_surcace_intersect=(coord_face[:,:,1]>=y_intersection).any(axis=1)*(coord_face[:,:,1]<=y_intersection).any(axis=1) # (m,)
  fig=go.Figure()
  fig.update_layout(scene=dict(aspectmode='data',
                               xaxis=dict(visible=False),
                               yaxis=dict(visible=False),
                               zaxis=dict(visible=False)),
                    margin=dict(l=0,r=0,b=0,t=0))
  c_surf=face[msk_face_surface]
  c_intersect=face[msk_surcace_intersect]
  fig.add_trace(go.Mesh3d(x=coord[:,0],y=coord[:,1],z=coord[:,2],i=c_surf[:,0],j=c_surf[:,1],k=c_surf[:,2],color='lightblue'))
  fig.add_trace(go.Mesh3d(x=coord[:,0],y=coord[:,1],z=coord[:,2],i=c_intersect[:,0],j=c_intersect[:,1],k=c_intersect[:,2],color='red'))
  return fig
