
//Description:
//Threshold a voxel dataset where we only extract the exterior faces and
//pass those faces to openGL for rendering

#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayHandleCounting.h>
#include <dax/Extent.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/CellTag.h>
#include <dax/CellTraits.h>

#include <dax/math/Compare.h> //Min && Max
#include <dax/math/Precision.h> //Floor
#include <dax/math/VectorAnalysis.h> //Normal && Cross

//exec headers we need
#include <dax/exec/internal/WorkletBase.h> //required for error handling
#include <dax/exec/CellVertices.h>

//headers we need for opengl support
#include <dax/opengl/testing/WindowBase.h>
#include <dax/opengl/TransferToOpenGL.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

namespace
{

GLvoid* bufferObjectPtr( unsigned int idx )
  {
  return (GLvoid*) ( ((char*)NULL) + idx );
  }

int grid_size = 256;
typedef dax::Tuple<dax::Vector3,4> QuadCoordType;
typedef dax::Tuple<dax::Vector3,4> QuadNormalType;
typedef dax::Tuple<dax::Vector4,4> QuadColorType;


//The functor used to determine if a single cell passes the threshold reqs
template<typename T>
struct threshold_voxel : public dax::exec::internal::WorkletBase
{
  //we inherit from WorkletBase so that we can throw errors in the exec
  //env and the control env can find out why the worklet failed

  typedef dax::CellTagVoxel CellTag;
  dax::exec::internal::TopologyUniform Topology; //holds the cell connectivity

  //hold a portal so that we can get values in the exec env
  typedef typename dax::cont::ArrayHandle< T >::PortalConstExecution PortalType;
  PortalType Values;

  //holds the array of of what cells pass or fail the threshold reqs
  typedef dax::cont::ArrayHandle< int >::PortalExecution OutPortalType;
  OutPortalType PassesThreshold;

  T MinValue;
  T MaxValue;

  DAX_CONT_EXPORT
  threshold_voxel(dax::cont::UniformGrid< > grid,
                 dax::cont::ArrayHandle<T> values,
                 T min, T max,
                 dax::cont::ArrayHandle<int> passes):
  Topology(grid.PrepareForInput()), //upload grid topology to exec env
  Values(values.PrepareForInput()), //upload values to exec env
  MinValue(min),
  MaxValue(max),
  PassesThreshold(passes.PrepareForOutput( grid.GetNumberOfCells() ))
    {
    }

  DAX_EXEC_EXPORT
  void operator()( int cell_index ) const
    {
    //get all the point ids for the cell index
    dax::exec::CellVertices<CellTag> verts =
                                  this->Topology.GetCellConnections(cell_index);

    //for each vertice see if we are between the min and max which is
    //inclusive on both sides. We hint to the compiler that this is
    //a fixed size array by using NUM_VERTICES. This can be easily
    //unrolled if needed
    int valid = 1;
    for(int i=0; i < dax::CellTraits<CellTag>::NUM_VERTICES; ++i)
      {
      const T value = this->Values.Get( verts[i] );
      valid &= value >= this->MinValue && value <= this->MaxValue;
      }
    this->PassesThreshold.Set(cell_index,valid);
    }
};

//this struct will count the number of cell neighbors that are valid
//based on that we can determine the number of faces that need to be shown
struct number_of_valid_neighhbors: public dax::exec::internal::WorkletBase
{
  //store the cell type that we are working on
  typedef dax::CellTagVoxel CellTag;

  //hold how many cells each cell will output
  typedef dax::cont::ArrayHandle< int >::PortalConstExecution PortalType;
  PortalType ValidCellIds;
  PortalType CellCountPortal;


  //holds if this cell should be renderd
  typedef dax::cont::ArrayHandle< bool >::PortalExecution OutPortalType;
  OutPortalType ShouldRender;

  dax::Id3 Dims;

  DAX_CONT_EXPORT
  number_of_valid_neighhbors(dax::cont::UniformGrid< > grid,
                 dax::cont::ArrayHandle<int> valid_cell_ids,
                 dax::cont::ArrayHandle<int> cell_counts,
                 dax::cont::ArrayHandle<bool> should_render):
  ValidCellIds(valid_cell_ids.PrepareForInput()),
  CellCountPortal(cell_counts.PrepareForInput()),
  ShouldRender(should_render.PrepareForOutput( valid_cell_ids.GetNumberOfValues() )),
  Dims(dax::extentCellDimensions(grid.GetExtent()))
    {
    }

  DAX_EXEC_EXPORT
  void operator()( int index ) const
    {
    //find the mapping from new cell id to original id
    int cell_index = this->ValidCellIds.Get(index);

    //compute the neighbors of this cell
    int neighbors[6] = { cell_index - this->Dims[0],
                         cell_index + 1,
                         cell_index + this->Dims[0],
                         cell_index - 1,
                         cell_index - this->Dims[0] * this->Dims[1],
                         cell_index + this->Dims[0] * this->Dims[1]};

    //determine if we are a cell that is on the boundary of any sides
    const int i = cell_index % this->Dims[0];
    const int j = (cell_index  / this->Dims[0]) % this->Dims[1];
    const int k = cell_index / (this->Dims[0] * this->Dims[1]);

    int neighbor_count = !(j == 0)            && (CellCountPortal.Get(neighbors[0]) > 0);
    neighbor_count += !(i == this->Dims[0]-1) && (CellCountPortal.Get(neighbors[1]) > 0);
    neighbor_count += !(j == this->Dims[1]-1) && (CellCountPortal.Get(neighbors[2]) > 0);
    neighbor_count += !(i == 0)               && (CellCountPortal.Get(neighbors[3]) > 0);
    neighbor_count += !(k == 0)               && (CellCountPortal.Get(neighbors[4]) > 0);
    neighbor_count += !(k == this->Dims[2]-1) && (CellCountPortal.Get(neighbors[5]) > 0);

    //set us to 1 if we are going to generate faces, otherwise set to 0
    this->ShouldRender.Set(index,(neighbor_count != 6));
    }
};

template<typename T>
struct make_faces : public dax::exec::internal::WorkletBase
{
  typedef dax::CellTagVoxel CellTag;

  //hold a portal so that we can get values in the exec env
  typedef typename dax::cont::ArrayHandle< T >::PortalConstExecution ValuesPortalType;
  typedef typename dax::cont::ArrayHandle< int >::PortalConstExecution IntPortalType;

  typedef dax::cont::ArrayHandle< QuadCoordType >::PortalExecution OutCoordPortalType;
  typedef dax::cont::ArrayHandle< QuadNormalType >::PortalExecution OutNormPortalType;
  typedef dax::cont::ArrayHandle< QuadColorType >::PortalExecution OutColorPortalType;

  typedef dax::exec::internal::TopologyUniform InTopologyType;
  typedef typename dax::cont::UniformGrid< >::PointCoordinatesType::PortalConstExecution Vector3PortalType;

  ValuesPortalType Values;
  IntPortalType ValidCellIds;

  OutCoordPortalType Coords;
  OutNormPortalType Norms;
  OutColorPortalType Colors;

  InTopologyType InputTopology; //holds the input cell connectivity
  Vector3PortalType InputCoords; //coordinates of the input voxel

  T MinValue;
  T MaxValue;

  DAX_CONT_EXPORT make_faces(dax::cont::UniformGrid< > grid,
                             dax::cont::ArrayHandle<T> values,
                             T min_value, T max_value,
                             const dax::cont::ArrayHandle<int>& valid_cell_ids,
                             dax::cont::ArrayHandle<QuadCoordType> coords,
                             dax::cont::ArrayHandle<QuadNormalType> norms,
                             dax::cont::ArrayHandle<QuadColorType> colors):
  Values(values.PrepareForInput()),
  ValidCellIds(valid_cell_ids.PrepareForInput()),
  Coords(coords.PrepareForOutput(valid_cell_ids.GetNumberOfValues()*6)),
  Norms(norms.PrepareForOutput(valid_cell_ids.GetNumberOfValues()*6)),
  Colors(colors.PrepareForOutput(valid_cell_ids.GetNumberOfValues()*6)),
  InputTopology(grid.PrepareForInput()),
  InputCoords(grid.GetPointCoordinates().PrepareForInput()),
  MinValue(min_value),
  MaxValue(max_value)
  {
  }

  DAX_EXEC_EXPORT
  void operator()( int index ) const
  {
    //we need to generate the following per voxel
    //1. 24 coordinates that describe the points of each quad
    //2. 6 normals one for each quad
    //3. 6 colors one for each quad

    const unsigned char vertices_for_faces[24] =
      {
       0, 1, 5, 4, // face 0
       1, 2, 6, 5, // face 1
       2, 3, 7, 6, // face 2
       0, 4, 7, 3, // face 3
       0, 3, 2, 1, // face 4
       4, 5, 6, 7  // face 5
      };

    //hsv color wheel computed 6 sections of r,g,b,a. This is done as a lookup
    //table to remove the branching that the origin version had
    //each of these values correspond to the color_values table we populate
    // with 0=Hue,1=p,2,=q,3=t
    dax::Vector4 color_values;
    const unsigned char reds[6]   = {0,2,1,1,3,0};
    const unsigned char greens[6] = {3,0,0,2,1,1};
    const unsigned char blues[6]  = {1,1,3,0,0,2};

    //find the mapping from new cell id to original id
    int cell_index = this->ValidCellIds.Get(index);

    dax::exec::CellVertices<CellTag> verts =
                            this->InputTopology.GetCellConnections(cell_index);
    dax::Tuple<dax::Vector3,8> in_coords;
    dax::Tuple<dax::Vector4,8> computed_point_colors;

    //compute the point ids for the voxel, and the colors for each point
    //of the voxel
    for(int i=0; i < 8; ++i)
      {
      in_coords[i] = this->InputCoords.Get(verts[i]);

      dax::Scalar interp =
        (this->Values.Get(verts[i]) - this->MinValue) / ( this->MaxValue - this->MinValue);
      interp = dax::math::Max(0.0f, dax::math::Min(1.0f,interp) );

      //h, s, v where h = interp * 5
      //s = 1.0f
      //v = 0.7f
      const float h = interp*5.0f;
      const float s = 1.0f;
      const float v = 0.7f;
      const unsigned int index = static_cast<unsigned int>(dax::math::Floor(h));
      const dax::Scalar fraction = h - index;

      color_values = dax::Vector4(v,
                                  v * (1.0f - s),
                                  v * (1.0f - s * fraction),
                                  v * (1.0f - s * (1-fraction)));

      computed_point_colors[i][0] = color_values[reds[index]];
      computed_point_colors[i][1] = color_values[greens[index]];
      computed_point_colors[i][2] = color_values[blues[index]];
      computed_point_colors[i][3] = 1.0f;
      }

    int write_pos = index * 6;
    for(int i=0; i < 24; i+=4, ++write_pos)
      {
        //write out the coordinates and normals per face
        {
        QuadCoordType out_coord;
        out_coord[0] = in_coords[vertices_for_faces[i]];
        out_coord[1] = in_coords[vertices_for_faces[i+1]];
        out_coord[2] = in_coords[vertices_for_faces[i+2]];
        out_coord[3] = in_coords[vertices_for_faces[i+3]];
        this->Coords.Set(write_pos,out_coord);

        QuadNormalType out_norms;
        const dax::Vector3 edge0 = out_coord[2] - out_coord[0];
        const dax::Vector3 edge1 = out_coord[3] - out_coord[1];
        const dax::Vector3 normal = dax::math::Normal( dax::math::Cross(edge0,edge1) );
        out_norms[0] = normal;
        out_norms[1] = normal;
        out_norms[2] = normal;
        out_norms[3] = normal;
        this->Norms.Set(write_pos,out_norms);
        }

        //write out colors for the face
        {
        QuadColorType out_color;
        out_color[0] = computed_point_colors[vertices_for_faces[i]];
        out_color[1] = computed_point_colors[vertices_for_faces[i+1]];
        out_color[2] = computed_point_colors[vertices_for_faces[i+2]];
        out_color[3] = computed_point_colors[vertices_for_faces[i+3]];

        this->Colors.Set(write_pos,out_color);
        }
      }
  }
};
}

//we could template this class on the vector type, but why bother for this example?
class threshold_renderer :
                  public dax::opengl::testing::WindowBase<threshold_renderer>
{
private:
  //find the default device adapter
  typedef DAX_DEFAULT_DEVICE_ADAPTER_TAG AdapterTag;

  //Make it easy to call the DeviceAdapter with the right tag
  typedef dax::cont::internal::DeviceAdapterAlgorithm<AdapterTag> DeviceAdapter;

  dax::cont::UniformGrid<> InputGrid;
  dax::cont::ArrayHandle<float> InputScalars;
  float MinValue;
  float MaxValue;
  int NumberOfFaces;

  bool ReRender;
  bool ReMesh;

  float MouseX, MouseY, RotateX, RotateY, TranslateZ;
  int ActiveMouseButtons;


  //gl array ids that hold the rendering info
  GLuint CoordHandle, ColorsHandle, NormalsHandle;

public:
  DAX_CONT_EXPORT threshold_renderer(const dax::cont::UniformGrid<>& grid,
                                     std::vector<float>& values,
                                     float minv, float maxv)
  {
  //lets get the threshold ready to rock and roll
  this->InputGrid = grid;
  //this doesn't copy the data it just references the vectors internal data
  this->InputScalars = dax::cont::make_ArrayHandle(values);
  this->MinValue = minv;
  this->MaxValue = maxv;
  this->NumberOfFaces = 0;

  this->ReMesh = true;
  this->ReRender = true;

  this->MouseX = 0;
  this->MouseY = 0;
  this->ActiveMouseButtons = 0;

  this->RotateX = 0;
  this->RotateY = 0;
  this->TranslateZ = 0;
  }

  void contstruct_render_data()
  {
  //schedule the thresholding on a per cell basis
  dax::cont::ArrayHandle<int> passesThreshold;
  threshold_voxel<float> tc(this->InputGrid,
                            this->InputScalars,
                            this->MinValue,
                            this->MaxValue,
                            passesThreshold);

  //since we are a uniform grid we can leverage the block scheduler in dax
  dax::Id3 dims = dax::extentCellDimensions(this->InputGrid.GetExtent());
  DeviceAdapter::Schedule( tc, dims );

  //extract only cells ids which pass the threshold
  dax::cont::ArrayHandle<int> validCellIndices;
  DeviceAdapter::StreamCompact( passesThreshold,
                                validCellIndices);
  if(validCellIndices.GetNumberOfValues() == 0)
    {
    this->NumberOfFaces = 0;
    this->ReMesh = false;
    return;
    }

  //now that we have the good cell ids only, lets
  //see which of those cells we want to generate faces for.
  dax::cont::ArrayHandle<bool> shouldRenderCell;
  number_of_valid_neighhbors nv(this->InputGrid,
                                validCellIndices,
                                passesThreshold,
                                shouldRenderCell);
  DeviceAdapter::Schedule( nv,  validCellIndices.GetNumberOfValues() );

  //compact again on cells that we need to render
  dax::cont::ArrayHandle<int> cellsToRender;
  DeviceAdapter::StreamCompact( validCellIndices,
                                shouldRenderCell,
                                cellsToRender);
  validCellIndices.ReleaseResources();
  shouldRenderCell.ReleaseResources();

  //we can now generate the faces for the cells
  dax::cont::ArrayHandle<QuadCoordType> coords;
  dax::cont::ArrayHandle<QuadNormalType> normals;
  dax::cont::ArrayHandle<QuadColorType> colors;

  make_faces<float> mf(this->InputGrid,
                       this->InputScalars,
                       this->MinValue,
                       this->MaxValue,
                       cellsToRender,
                       coords,
                       normals,
                       colors);

  DeviceAdapter::Schedule(mf, cellsToRender.GetNumberOfValues() );

  //now push that to opengl
  dax::opengl::TransferToOpenGL(coords, this->CoordHandle);
  dax::opengl::TransferToOpenGL(normals, this->NormalsHandle);
  dax::opengl::TransferToOpenGL(colors, this->ColorsHandle);

  this->NumberOfFaces = 6 * cellsToRender.GetNumberOfValues();
  this->ReMesh = false;

  std::cout << "done making " << this->NumberOfFaces << " faces" << std::endl;
  }

  //called after opengl is inited, this is where we can set our gl flags
  DAX_CONT_EXPORT void PostInit()
  {
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);

    // good old-fashioned fixed function lighting
    float white[4] = { 0.8, 0.8, 0.8, 1.0 };
    float lightPos[4] = { 100.0, 100.0, -100.0, 1.0 };

    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 100);

    glLightfv(GL_LIGHT0, GL_AMBIENT, white);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
    glLightfv(GL_LIGHT0, GL_SPECULAR, white);
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);

    // Setup the view of the cube.
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective( 60.0f, 1.0, 1.0, grid_size*4.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt((grid_size-1)/2, (grid_size-1)/2, grid_size*1.5,
              (grid_size-1)/2, (grid_size-1)/2, 0.0,
              0.0, 1.0, 0.0);

    this->contstruct_render_data();
    this->ReRender = true;
  }

  DAX_CONT_EXPORT void Display()
  {
    if(this->ReMesh) { this->contstruct_render_data();}
    else if((!this->ReRender) || this->NumberOfFaces == 0) {return;}

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // set view matrix for 3D scene
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    //move the camera
    glTranslatef(0.0, 0.0, this->TranslateZ);
    glRotatef(this->RotateX, 1.0, 0.0, 0.0);
    glRotatef(this->RotateY, 0.0, 1.0, 0.0);


    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, this->CoordHandle);
    glVertexPointer(3, GL_FLOAT, 0, bufferObjectPtr(0) );

    glEnableClientState(GL_COLOR_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, this->ColorsHandle);
    glColorPointer(4, GL_FLOAT, 0, bufferObjectPtr(0) );

    glEnableClientState(GL_NORMAL_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, this->NormalsHandle);
    glNormalPointer(GL_FLOAT, 0, bufferObjectPtr(0) );

    glDrawArrays( GL_QUADS, 0,  this->NumberOfFaces );

    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);


    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glPopMatrix();

    //quick and ReRender swapping
    glutSwapBuffers();
    this->ReRender = false;
  }

  DAX_CONT_EXPORT void Idle(){ glutPostRedisplay(); }

  DAX_CONT_EXPORT void ChangeSize(int daxNotUsed(w), int daxNotUsed(h)) {}

  DAX_CONT_EXPORT void Key(unsigned char daxNotUsed(key), int daxNotUsed(x),
                           int daxNotUsed(y) ){ }

  DAX_CONT_EXPORT void SpecialKey(int key,
                                  int daxNotUsed(x), int daxNotUsed(y) )
  {
  switch (key)
    {
    case GLUT_KEY_UP:
      this->MinValue = dax::math::Min(this->MinValue+5,this->MaxValue-1);
      this->ReMesh = true;
      break;
    case GLUT_KEY_DOWN :
      this->MinValue = dax::math::Max(this->MinValue-5,1.0f);
      this->ReMesh = true;
      break;
    default:
      break;
    }
  }

  DAX_CONT_EXPORT void Mouse(int button, int state,
                             int x, int y )
  {
    if (state == GLUT_DOWN)
    {
    this->ActiveMouseButtons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
    this->ActiveMouseButtons = 0;
    }

    this->MouseX = x;
    this->MouseY = y;
  }

  DAX_CONT_EXPORT void MouseMove(int x, int y )
  {
    float dx = (float)(x - MouseX);
    float dy = (float)(y - MouseY);

    if (ActiveMouseButtons & 1)
    {
        this->ReRender = true;
        this->RotateX += dy * 0.4f;
        this->RotateY += dx * 0.4f;
    }
    else if (ActiveMouseButtons & 4)
    {
        this->ReRender = true;
        this->TranslateZ += dy * 0.4f;
    }

    this->MouseX = x;
    this->MouseY = y;
  }
};

struct float_rand
{

  float operator()() const
  {
    return (std::rand() % 10000) /42.0f;
  }
};

int main(int, char**)
{
  dax::cont::UniformGrid<> grid;
  grid.SetExtent( dax::Id3(0,0,0),
                  dax::Id3(grid_size-1,grid_size-1,grid_size-1));

  std::vector<float> data_store(grid_size*grid_size*grid_size);

  //fill the vector with random numbers
  std::srand(42); //I like this seed :-)
  std::generate(data_store.begin(),data_store.end(),float_rand());

  const float sum = std::accumulate(data_store.begin(),data_store.end(),0.0f);
  const float average = sum / static_cast<float>(data_store.size());
  const float max = *(std::max_element(data_store.begin(),data_store.end()));

  std::cout << "min: " << average << " max: " << max << std::endl;

  //setup the input to threshold
  threshold_renderer renderer(grid,data_store,average,max);

  //init the render window
  renderer.Init("Threshold Example", 800, 600);

  //start drawing
  renderer.Start();
  return 0;
}
