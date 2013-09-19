
//Description:
//Threshold a voxel dataset where we only extract the exterior faces and
//pass those faces to openGL for rendering

#include <dax/cont/Scheduler.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayHandleCounting.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/CellTag.h>
#include <dax/CellTraits.h>

#include <dax/math/Compare.h> //Min && Max
#include <dax/math/Precision.h> //Floor
#include <dax/math/Trig.h> //Sin Cos, etc
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
typedef dax::Tuple<unsigned char,4> ColorType;

class InitFields : public dax::exec::WorkletMapField
{
  dax::Id XDim, YDim;
public:
  typedef void ControlSignature(Field(In), Field(In), Field(In), Field(Out), Field(Out), Field(Out));
  typedef void ExecutionSignature(_1, _2, _3, _4, _5, _6);

  InitFields(dax::Id mesh_x_dim, dax::Id mesh_y_dim):
    XDim(mesh_x_dim), YDim(mesh_y_dim)
    {}

  DAX_EXEC_EXPORT
  void operator()(dax::Id index,
                  dax::Scalar frequency,
                  dax::Vector2 seeds,
                  dax::Vector3& coord,
                  ColorType& color,
                  dax::Vector2& velocity ) const
  {
    //compute our places given the flat index
    const dax::Id x_index = index % XDim;
    const dax::Id y_index = index / YDim;

    //compute out our coords
    coord[0] = x_index / static_cast<dax::Scalar>(XDim) + seeds[0];
    coord[1] = y_index / static_cast<dax::Scalar>(YDim) + seeds[1];
    coord[2] = dax::math::Sin(coord[0]*frequency) *
               dax::math::Cos(coord[1]*frequency) * 0.2f;

    //compute the starting color and velcity
    color=ColorType(0,255,255,1);
    velocity=dax::make_Vector2(0.0f,0.0f);

  }
};

class ComputeStep : public dax::exec::WorkletMapField
{
  dax::Scalar TimeStep;
  dax::Id XDim, YDim;
  dax::Scalar MouseX, MouseY;
public:
  typedef void ControlSignature(Field(In), Field(In), Field(In), Field(In), Field(Out), Field(Out), Field(Out));
  typedef void ExecutionSignature(_1, _2, _3, _4, _5, _6, _7);

  ComputeStep(dax::Scalar time, dax::Id mesh_x_dim, dax::Id mesh_y_dim,
              dax::Scalar mouse_x, dax::Scalar mouse_y):
    TimeStep(time),XDim(mesh_x_dim), YDim(mesh_y_dim),
    MouseX(mouse_x), MouseY(mouse_y)
    {}

  DAX_EXEC_EXPORT
  void operator()(dax::Id index,
                  dax::Scalar frequency,
                  const dax::Vector3& in_pos,
                  const dax::Vector2& in_vel,
                  dax::Vector3& pos,
                  ColorType& color,
                  dax::Vector2& vel ) const
  {
    const dax::Scalar speed = 0.0005f;

    const dax::Id x_index = index % XDim;
    const dax::Id y_index = index / YDim;

    const dax::Scalar xX = (this->MouseX - XDim/2 + 128)/(float)XDim*4.5f;
    const dax::Scalar yY = (this->MouseY - YDim/2 + 128)/(float)YDim*4.5f;
    const dax::Scalar u = x_index / (float) XDim;
    const dax::Scalar v = y_index / (float) YDim;

    const dax::Vector2 computed_xyz(-in_pos[0] + xX, -in_pos[2] + yY);
    const dax::Vector2 normalized = dax::math::Normal(computed_xyz);
    vel = in_vel + normalized;

    const dax::Scalar updated_speed = dax::math::Sqrt(dax::dot(vel,vel));
    const dax::Scalar original_speed = dax::math::Sqrt(dax::dot(computed_xyz,computed_xyz));


    if(updated_speed >= 0.125f)
      {
      vel[0] = normalized[0]/original_speed*0.1f;
      vel[1] = normalized[1]/original_speed*0.1f;
      }

    pos[0] = in_pos[0] + vel[0] * updated_speed;
    pos[1] = dax::math::Sin(u*frequency + TimeStep) *
             dax::math::Cos(v*frequency + TimeStep) * 0.2f;
    pos[2] = in_pos[2] + vel[1] * updated_speed;

    color = ColorType(128/original_speed,(int)(255/(original_speed*51)),255,10);

  }
};

struct TwizzledGLHandles
{
  //stores two sets of handles, each time
  //you call for the handles it switches
  //what ones it returns
  TwizzledGLHandles()
    {
    this->State = 0;
    }

  void initHandles()
  {
    glGenBuffers(2,CoordHandles);
    glGenBuffers(2,ColorHandles);

    glBindBuffer(GL_ARRAY_BUFFER, CoordHandles[0]);
    glBindBuffer(GL_ARRAY_BUFFER, CoordHandles[1]);
    glBindBuffer(GL_ARRAY_BUFFER, ColorHandles[0]);
    glBindBuffer(GL_ARRAY_BUFFER, ColorHandles[1]);
  }

  void handles(GLuint& coord, GLuint& color) const
    {
    coord = CoordHandles[this->State%2];
    color = ColorHandles[this->State%2];
    }

  void switchHandles()
    {
    ++this->State;
    }
private:
  GLuint CoordHandles[2];
  GLuint ColorHandles[2];
  unsigned int State; //when we roll over we need to stay positive
};


class particle_renderer :
                  public dax::opengl::testing::WindowBase<particle_renderer>
{
private:
  GLvoid* bufferObjectPtr( unsigned int idx )
    {
    return (GLvoid*) ( ((char*)NULL) + idx );
    }

  dax::cont::Scheduler<> Scheduler;
  dax::cont::ArrayHandle<dax::Vector2> SeedHandle;
  dax::cont::ArrayHandle<dax::Vector3> ParticleCoords;
  dax::cont::ArrayHandle<dax::Vector2> ParticleVelocity;
  dax::cont::ArrayHandle<ColorType> ParticleColors;

  unsigned int XDim, YDim;
  dax::Scalar TimeStep;
  dax::Scalar Frequency;

  float MouseX, MouseY, RotateX, RotateY, TranslateZ;
  int ActiveMouseButtons;

  //gl array ids that hold the rendering info
  TwizzledGLHandles TwizzleHandles;

public:
  DAX_CONT_EXPORT particle_renderer( unsigned int x_dim,
                                     unsigned int y_dim,
                                     std::vector<dax::Vector2>& seeds)
  {
  //this doesn't copy the data it just references the vectors internal data
  this->XDim = x_dim;
  this->YDim = y_dim;
  this->SeedHandle = dax::cont::make_ArrayHandle(seeds);

  this->MouseX = 0;
  this->MouseY = 0;
  this->ActiveMouseButtons = 0;

  this->TimeStep = 0.0f;
  this->Frequency = 2.42f;

  this->RotateX = 0;
  this->RotateY = 0;
  this->TranslateZ = -3.0;
  }

  void construct_starting_data()
  {
    GLuint coord, color;
    this->TwizzleHandles.handles(coord,color);

    //pass the initial seed data
    InitFields init(this->XDim,this->YDim);
    const dax::Id size = this->XDim * this->YDim;


    this->Scheduler.Invoke(init,
                           dax::cont::make_ArrayHandleCounting(0,size),
                           this->Frequency,
                           this->SeedHandle,
                           this->ParticleCoords,
                           this->ParticleColors,
                           this->ParticleVelocity);

    this->SeedHandle.ReleaseResourcesExecution();
    //now push that to opengl
    dax::opengl::TransferToOpenGL(this->ParticleCoords, coord);
    dax::opengl::TransferToOpenGL(this->ParticleColors, color);


    this->TwizzleHandles.switchHandles();
    this->TwizzleHandles.handles(coord,color);
    dax::opengl::TransferToOpenGL(this->ParticleCoords, coord);
    dax::opengl::TransferToOpenGL(this->ParticleColors, color);


  }

  void compute_next_step()
  {
    GLuint coord, color;
    this->TwizzleHandles.handles(coord,color);

    ComputeStep compu_step(this->TimeStep, this->XDim, this->YDim,
                           this->MouseX, this->MouseY);
    const dax::Id size = this->XDim * this->YDim;

    this->Scheduler.Invoke(compu_step,
                           dax::cont::make_ArrayHandleCounting(0,size),
                           this->Frequency,
                           this->ParticleCoords,
                           this->ParticleVelocity,
                           this->ParticleCoords,
                           this->ParticleColors,
                           this->ParticleVelocity);

    dax::opengl::TransferToOpenGL(this->ParticleCoords, coord);
    dax::opengl::TransferToOpenGL(this->ParticleColors, color);
  }

  //called after opengl is inited, this is where we can set our gl flags
  DAX_CONT_EXPORT void PostInit()
  {
    //init the gl handles for the twizzler
    this->TwizzleHandles.initHandles();

    glDisable(GL_DEPTH_TEST);
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

    glClearColor(0.0, 0.0, 0.0, 1.0);
    //Viewport
    glViewport(0, 0, 800.0f, 600.0f);

    //Projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (800.0f /  600.0f), 0.1, 10.0);

    this->construct_starting_data();
  }

  DAX_CONT_EXPORT void Display()
  {
    GLuint coord, color;
    this->TwizzleHandles.handles(coord,color);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //View matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glTranslatef(0.0, 0.0, this->TranslateZ);
    glRotatef(90.0, 1.0, 0.0, 0.0);

    //Render from VBO
    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, coord);
    glVertexPointer(3, GL_FLOAT, 0, bufferObjectPtr(0) );

    glEnableClientState(GL_COLOR_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, color);
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, bufferObjectPtr(0) );


    const dax::Id size = this->XDim * this->YDim;

    glDrawArrays(GL_POINTS, 0, size);

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);


    glutSwapBuffers();
    glutPostRedisplay();

    this->TimeStep += 0.01;

    this->TwizzleHandles.switchHandles();
    this->compute_next_step();
  }

  DAX_CONT_EXPORT void Idle(){ }

  DAX_CONT_EXPORT void ChangeSize(int daxNotUsed(w), int daxNotUsed(h)) {}

  DAX_CONT_EXPORT void Key(unsigned char key, int daxNotUsed(x), int daxNotUsed(y) )
  {
    switch(key)
      {
      case(27):
        exit(0);
      default:
        break;
      }
  }

  DAX_CONT_EXPORT void SpecialKey(int key,
                                  int daxNotUsed(x), int daxNotUsed(y) )
  {
  }

  DAX_CONT_EXPORT void Mouse(int button, int state,
                             int x, int y )
  {
  }

  DAX_CONT_EXPORT void MouseMove(int x, int y )
  {
    this->MouseX = x;
    this->MouseY = y;
  }
};

struct seed_rand
{
  dax::Vector2 operator()() const
  {
    dax::Vector2 vec;
    vec[0] = ((std::rand() % 100) - 100) / 2400.0f;
    vec[1] = ((std::rand() % 100) - 100) / 2400.0f;
    return vec;
  }
};
}

int main(int, char**)
{
  const unsigned int x_dim = 1024;
  const unsigned int y_dim = 1024;
  std::vector<dax::Vector2> seeds(x_dim*y_dim);

  //fill the vector with random numbers
  std::srand(42); //I like this seed :-)
  std::generate(seeds.begin(),seeds.end(),seed_rand());

  //setup the input to threshold
  particle_renderer renderer(x_dim,y_dim,seeds);

  //init the render window
  renderer.Init("Particle Demo", 800, 600);

  //start drawing
  renderer.Start();
  return 0;
}
