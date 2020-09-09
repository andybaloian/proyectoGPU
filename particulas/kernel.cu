
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/shader_s.h>
#include <learnopengl/camera.h>

#include <iostream>
#include <map>
#include <string>
#include <random>
#include <iomanip>
#include <sstream>
#include <math.h>

#include <ft2build.h>
#include FT_FREETYPE_H  

const int na = 36;        // vertex grid size
const int nb = 18;
const int na3 = na * 3;     // line in grid size
const int nn = nb * na3;    // whole grid size
const int n_particulas = 1000; //numero de particulas
const float radio = 0.03f; //radio de las esferas

//parametros fisicos
float gravedad = 10.0f;
float roce = 0.4f;

//actualizacion fps
float periodo_fps = 1.0f;
float intervalo_fps = 0.0f;
float fps = 0.0f;
int count_fps = 0.0;

const float M_PI = 3.14;

//cuda

cudaError_t choques_cuda(float* posiciones_anteriores, float* nuevas_posiciones, float* velocidades_anteriores, float* nuevas_velocidades);

__device__ float distancia(float* a, float* b) {
    return sqrt(pow((float)(a[0] - b[0]), (float)2) + pow((float)(a[1] - b[1]), (float)2) + pow((float)(a[2] - b[2]), (float)2));
}

__device__ float modulo(float* v) {
    return sqrt(pow((float)(v[0]), (float)2) + pow((float)(v[1]), (float)2) + pow((float)(v[2]), (float)2));
}

__global__ void kernel_choques(int particulas, float diametro, float* posiciones_anteriores, float* nuevas_posiciones, float* velocidades_anteriores, float* nuevas_velocidades)
{
    
    int indice_particula = threadIdx.x + blockIdx.x * blockDim.x;
    //printf("thread de particula %i\n", indice_particula);
    //printf("--posicion de particula %i: %f, %f, %f \n", indice_particula, posiciones[indice_particula * 3], posiciones[indice_particula * 3 + 1], posiciones[indice_particula * 3 + 2]);
    //printf("--velocidad de particula %i: %f, %f, %f\n", indice_particula, velocidades_anteriores[indice_particula * 3], velocidades_anteriores[indice_particula * 3 + 1], velocidades_anteriores[indice_particula * 3 + 2]);
    bool choque = false;
    //printf("posiciones: %f, %f, %f, %f, %f, %f \n", posiciones[0], posiciones[1], posiciones[2], posiciones[3], posiciones[4], posiciones[5]);
    for (int i = 0; i < particulas; i++) {
        //printf("%i", i);
        //printf("Velocidad de particula %i: %f, %f, %f\n", indice_particula, velocidades_anteriores[indice_particula], velocidades_anteriores[indice_particula + 1], velocidades_anteriores[indice_particula + 2]);
        if (i == indice_particula) {
            continue;
        }
        float v1[3] = {
            posiciones_anteriores[i * 3], posiciones_anteriores[i * 3 + 1], posiciones_anteriores[i * 3 + 2]
        };

        float v2[3] = {
            posiciones_anteriores[indice_particula * 3], posiciones_anteriores[indice_particula * 3 + 1], posiciones_anteriores[indice_particula * 3 + 2]
        };

        float dist = distancia(v1, v2);
        //printf("    distancia entra particula %i y particula %i =  %f\n", indice_particula, i, dist);
        if (dist <= diametro) {
            //printf("    - choque de particula %i con particula %i\n", indice_particula, i);
            //printf("    - distancia =  %f\n", dist);
            float normal[] = { 0.0f, 0.0f, 0.0f };
            normal[0] = posiciones_anteriores[i * 3] - posiciones_anteriores[3 * indice_particula];
            normal[1] = posiciones_anteriores[i * 3 + 1] - posiciones_anteriores[3 * indice_particula + 1];
            normal[2] = posiciones_anteriores[i * 3 + 2] - posiciones_anteriores[3 * indice_particula + 2];
            float mod = modulo(normal);
            normal[0] = normal[0] / mod;
            normal[1] = normal[1] / mod;
            normal[2] = normal[2] / mod;

            float rel_vel[] = {
                velocidades_anteriores[i * 3] - velocidades_anteriores[3 * indice_particula],
                velocidades_anteriores[i * 3 + 1] - velocidades_anteriores[3 * indice_particula + 1],
                velocidades_anteriores[i * 3 + 2] - velocidades_anteriores[3 * indice_particula + 2]
            };

            float punto = normal[0] * rel_vel[0] + normal[1] * rel_vel[1] + normal[2] * rel_vel[2];

            float normal_vel[] = {
                normal[0] * punto,
                normal[1] * punto,
                normal[2] * punto
            };

            float solapamiento = diametro - dist;

            nuevas_velocidades[3 * indice_particula] = velocidades_anteriores[3 * indice_particula] + normal_vel[0];
            nuevas_velocidades[3 * indice_particula + 1] = velocidades_anteriores[3 * indice_particula + 1] + normal_vel[1];
            nuevas_velocidades[3 * indice_particula + 2] = velocidades_anteriores[3 * indice_particula + 2] + normal_vel[2];

            nuevas_posiciones[3 * indice_particula] = posiciones_anteriores[3 * indice_particula] - solapamiento / 2 * normal[0];
            nuevas_posiciones[3 * indice_particula + 1] = posiciones_anteriores[3 * indice_particula + 1] - solapamiento / 2 * normal[1];
            nuevas_posiciones[3 * indice_particula + 2] = posiciones_anteriores[3 * indice_particula + 2] - solapamiento / 2 * normal[2];

            choque = true;
            break;
        }
    }
    if (!choque) {
        nuevas_velocidades[3 * indice_particula] = velocidades_anteriores[3 * indice_particula];
        nuevas_velocidades[3 * indice_particula + 1] = velocidades_anteriores[3 * indice_particula + 1];
        nuevas_velocidades[3 * indice_particula + 2] = velocidades_anteriores[3 * indice_particula + 2];
        nuevas_posiciones[3 * indice_particula] = posiciones_anteriores[3 * indice_particula];
        nuevas_posiciones[3 * indice_particula + 1] = posiciones_anteriores[3 * indice_particula + 1];
        nuevas_posiciones[3 * indice_particula + 2] = posiciones_anteriores[3 * indice_particula + 2];
    }
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t choques_cuda(float* posiciones_anteriores, float* nuevas_posiciones, float* velocidades_anteriores, float* nuevas_velocidades)
{
    float* dev_posiciones_anteriores;
    float* dev_nuevas_posiciones;
    float* dev_velocidades_anteriores;
    float* dev_nuevas_velocidades;
    float diametro = 2 * radio;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_posiciones_anteriores, n_particulas * sizeof(float) * 3);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_nuevas_posiciones, n_particulas * sizeof(float) * 3);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_velocidades_anteriores, n_particulas * sizeof(float) * 3);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_nuevas_velocidades, n_particulas * sizeof(float) * 3);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_nuevas_posiciones, nuevas_posiciones, n_particulas * sizeof(float) * 3, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_posiciones_anteriores, posiciones_anteriores, n_particulas * sizeof(float) * 3, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_velocidades_anteriores, velocidades_anteriores, n_particulas * sizeof(float) * 3, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_nuevas_velocidades, nuevas_velocidades, n_particulas * sizeof(float) * 3, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    // Launch a kernel on the GPU with one thread for each element.

    kernel_choques << <1, n_particulas >> > (n_particulas, diametro, dev_posiciones_anteriores, dev_nuevas_posiciones, dev_velocidades_anteriores, dev_nuevas_velocidades);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(nuevas_posiciones, dev_nuevas_posiciones, n_particulas * sizeof(float) * 3, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(nuevas_velocidades, dev_nuevas_velocidades, n_particulas * sizeof(float) * 3, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_posiciones_anteriores);
    cudaFree(dev_nuevas_posiciones);
    cudaFree(dev_velocidades_anteriores);
    cudaFree(dev_nuevas_velocidades);

    return cudaStatus;
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;

GLfloat sphere_pos[nn]; // vertex
GLfloat sphere_nor[nn]; // normal
//GLfloat sphere_col[nn];   // color
GLuint  sphere_ix[na * (nb - 1) * 6];    // indices
GLuint sphere_vbo[4] = { -1,-1,-1,-1 };
GLuint sphere_vao[4] = { -1,-1,-1,-1 };

float get_random()
{
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(-0.5, 0.5); // rage 0 - 1
    return dis(e);
}

void sphere_init()
{
    // generate the sphere data
    GLfloat x, y, z, a, b, da, db, r = radio;
    int ia, ib, ix, iy;
    da = 2.0 * M_PI / GLfloat(na);
    db = M_PI / GLfloat(nb - 1);
    // [Generate sphere point data]
    // spherical angles a,b covering whole sphere surface
    for (ix = 0, b = -0.5 * M_PI, ib = 0; ib < nb; ib++, b += db)
        for (a = 0.0, ia = 0; ia < na; ia++, a += da, ix += 3)
        {
            // unit sphere
            x = cos(b) * cos(a);
            y = cos(b) * sin(a);
            z = sin(b);
            sphere_pos[ix + 0] = x * r;
            sphere_pos[ix + 1] = y * r;
            sphere_pos[ix + 2] = z * r;
            sphere_nor[ix + 0] = x;
            sphere_nor[ix + 1] = y;
            sphere_nor[ix + 2] = z;
        }
    // [Generate GL_TRIANGLE indices]
    for (ix = 0, iy = 0, ib = 1; ib < nb; ib++)
    {
        for (ia = 1; ia < na; ia++, iy++)
        {
            // first half of QUAD
            sphere_ix[ix] = iy;      ix++;
            sphere_ix[ix] = iy + 1;    ix++;
            sphere_ix[ix] = iy + na;   ix++;
            // second half of QUAD
            sphere_ix[ix] = iy + na;   ix++;
            sphere_ix[ix] = iy + 1;    ix++;
            sphere_ix[ix] = iy + na + 1; ix++;
        }
        // first half of QUAD
        sphere_ix[ix] = iy;       ix++;
        sphere_ix[ix] = iy + 1 - na;  ix++;
        sphere_ix[ix] = iy + na;    ix++;
        // second half of QUAD
        sphere_ix[ix] = iy + na;    ix++;
        sphere_ix[ix] = iy - na + 1;  ix++;
        sphere_ix[ix] = iy + 1;     ix++;
        iy++;
    }

    // [VAO/VBO stuff]
    GLuint i;
    glGenVertexArrays(4, sphere_vao);
    glGenBuffers(4, sphere_vbo);
    glBindVertexArray(sphere_vao[0]);
    i = 0; // vertex
    glBindBuffer(GL_ARRAY_BUFFER, sphere_vbo[i]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_pos), sphere_pos, GL_STATIC_DRAW);
    glEnableVertexAttribArray(i);
    glVertexAttribPointer(i, 3, GL_FLOAT, GL_FALSE, 0, 0);
    i = 1; // indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_vbo[i]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(sphere_ix), sphere_ix, GL_STATIC_DRAW);
    glEnableVertexAttribArray(i);
    glVertexAttribPointer(i, 4, GL_UNSIGNED_INT, GL_FALSE, 0, 0);
    i = 2; // normal
    glBindBuffer(GL_ARRAY_BUFFER, sphere_vbo[i]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(sphere_nor), sphere_nor, GL_STATIC_DRAW);
    glEnableVertexAttribArray(i);
    glVertexAttribPointer(i, 3, GL_FLOAT, GL_FALSE, 0, 0);
    /*
        i=3; // color
        glBindBuffer(GL_ARRAY_BUFFER,sphere_vbo[i]);
        glBufferData(GL_ARRAY_BUFFER,sizeof(sphere_col),sphere_col,GL_STATIC_DRAW);
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i,3,GL_FLOAT,GL_FALSE,0,0);
    */
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
}
void sphere_exit()
{
    glDeleteVertexArrays(4, sphere_vao);
    glDeleteBuffers(4, sphere_vbo);
}
void sphere_draw()
{
    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CCW);

    glBindVertexArray(sphere_vao[0]);
    //  glDrawArrays(GL_POINTS,0,sizeof(sphere_pos)/sizeof(GLfloat));                   // POINTS ... no indices for debug
    glDrawElements(GL_TRIANGLES, sizeof(sphere_ix) / sizeof(GLuint), GL_UNSIGNED_INT, 0);    // indices (choose just one line not both !!!)
    glBindVertexArray(0);
}

void RenderText(Shader& shader, std::string text, float x, float y, float scale, glm::vec3 color);

struct Character {
    unsigned int TextureID; // ID handle of the glyph texture
    glm::ivec2   Size;      // Size of glyph
    glm::ivec2   Bearing;   // Offset from baseline to left/top of glyph
    unsigned int Advance;   // Horizontal offset to advance to next glyph
};

std::map<GLchar, Character> Characters;
unsigned int VAO_text, VBO_text;

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    sphere_init();

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // build and compile our shader zprogram
    // ------------------------------------
    Shader ourShader("shader.vs", "shader.fs");

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------


    float wireframe_vertices[] = {
        -1.0f, -1.0f, 1.0f,
         1.0f, -1.0f, 1.0f,

         1.0f, -1.0f, 1.0f,
         1.0f,  1.0f, 1.0f,

         1.0f,  1.0f, 1.0f,
        -1.0f, 1.0f, 1.0f,

        -1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 1.0f,

        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,

         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,

         1.0f,  1.0f, -1.0f,
        -1.0f, 1.0f, -1.0f,

        -1.0f, 1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f, 1.0f,
        -1.0f, -1.0f, -1.0f,

         1.0f, -1.0f, 1.0f,
         1.0f, -1.0f, -1.0f,

         1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, -1.0f,

        -1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, -1.0f

    };

    glm::vec3 cubePositions[n_particulas];
    glm::vec3 cubeVelocities[n_particulas];

    
    for (int i = 0; i < n_particulas; i++) {
        cubePositions[i] = glm::vec3(get_random(), get_random(), get_random());
        cubeVelocities[i] = glm::vec3(get_random(), get_random(), get_random());
    }
    
    /*
    cubePositions[0] = glm::vec3(0.2f, -1.0f + radio, 0.0f);
    cubePositions[1] = glm::vec3(-0.2f, -1.0f + radio, 0.0f);
    cubeVelocities[0] = glm::vec3(-0.2f, 0.0f, 0.0f);
    cubeVelocities[1] = glm::vec3(0.0f, 0.0f, 0.0f);
    */

    //cubo wireframe

    unsigned int VBO_wireframe, VAO_wireframe;
    glGenVertexArrays(1, &VAO_wireframe);
    glGenBuffers(1, &VBO_wireframe);

    glBindVertexArray(VAO_wireframe);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_wireframe);
    glBufferData(GL_ARRAY_BUFFER, sizeof(wireframe_vertices), wireframe_vertices, GL_DYNAMIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    ourShader.use();

    // compile and setup the shader
    // ----------------------------
    Shader shader("text.vs", "text.fs");
    glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(SCR_WIDTH), 0.0f, static_cast<float>(SCR_HEIGHT));
    shader.use();
    glUniformMatrix4fv(glGetUniformLocation(shader.ID, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    // FreeType
    // --------
    FT_Library ft;
    // All functions return a value different than 0 whenever an error occurred
    if (FT_Init_FreeType(&ft))
    {
        std::cout << "ERROR::FREETYPE: Could not init FreeType Library" << std::endl;
        return -1;
    }

    // find path to font
    std::string font_name = "bahnschrift.ttf";
    if (font_name.empty())
    {
        std::cout << "ERROR::FREETYPE: Failed to load font_name" << std::endl;
        return -1;
    }

    // load font as face
    FT_Face face;
    if (FT_New_Face(ft, font_name.c_str(), 0, &face)) {
        std::cout << "ERROR::FREETYPE: Failed to load font" << std::endl;
        return -1;
    }
    else {
        // set size to load glyphs as
        FT_Set_Pixel_Sizes(face, 0, 48);

        // disable byte-alignment restriction
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        // load first 128 characters of ASCII set
        for (unsigned char c = 0; c < 128; c++)
        {
            // Load character glyph 
            if (FT_Load_Char(face, c, FT_LOAD_RENDER))
            {
                std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
                continue;
            }
            // generate texture
            unsigned int texture;
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RED,
                face->glyph->bitmap.width,
                face->glyph->bitmap.rows,
                0,
                GL_RED,
                GL_UNSIGNED_BYTE,
                face->glyph->bitmap.buffer
            );
            // set texture options
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            // now store character for later use
            Character character = {
                texture,
                glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
                glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
                static_cast<unsigned int>(face->glyph->advance.x)
            };
            Characters.insert(std::pair<char, Character>(c, character));
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    // destroy FreeType once we're finished
    FT_Done_Face(face);
    FT_Done_FreeType(ft);

    // configure VAO/VBO for texture quads
    // -----------------------------------
    glGenVertexArrays(1, &VAO_text);
    glGenBuffers(1, &VBO_text);
    glBindVertexArray(VAO_text);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_text);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        // --------------------
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // activate shader
        ourShader.use();

        // pass projection matrix to shader (note that in this case it could change every frame)
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        ourShader.setMat4("projection", projection);

        // camera/view transformation
        glm::mat4 view = camera.GetViewMatrix();
        ourShader.setMat4("view", view);

        //render cubo wireframe
        glBindVertexArray(VAO_wireframe);

        glm::mat4 model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
        ourShader.setMat4("model", model);
        ourShader.setVec4("color", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));

        glLineWidth(3.3f);
        glDrawArrays(GL_LINES, 0, 24);
        glLineWidth(1.0f);

        //procesar choques en cpu
        /*
        for (int i = 0; i < n_particulas; i++) {
            //choque en cpu (n^2)
            for (int j = i + 1; j < n_particulas; j++) {
                float dist = glm::distance(cubePositions[i], cubePositions[j]);
                if (dist < 2 * radio) {
                    glm::vec3 normal = glm::normalize(cubePositions[i] - cubePositions[j]);

                    //evitar que las esferas se solapen
                    float solapamiento = 2 * radio - dist;

                    cubePositions[i] = cubePositions[i] + solapamiento / 2 * normal;
                    cubePositions[j] = cubePositions[j] - solapamiento / 2 * normal;

                    glm::vec3 rel_vel = cubeVelocities[i] - cubeVelocities[j];
                    glm::vec3 normal_vel = glm::dot(rel_vel, normal) * normal;
                    //normal_vel = glm::vec3(0.0f, 0.0f, 0.0f);
                    cubeVelocities[i] = cubeVelocities[i] - normal_vel;
                    cubeVelocities[j] = cubeVelocities[j] + normal_vel;
                }
            }
        }
        */
        //procesar choques en gpu

        float posiciones_anteriores_arr[n_particulas * 3];
        float nuevas_posiciones_arr[n_particulas * 3];
        float velocidades_arr[n_particulas * 3];
        float nuevas_velocidades_arr[n_particulas * 3];

        for (int i = 0; i < n_particulas; i++) {

            posiciones_anteriores_arr[3 * i] = cubePositions[i][0];
            posiciones_anteriores_arr[3 * i + 1] = cubePositions[i][1];
            posiciones_anteriores_arr[3 * i + 2] = cubePositions[i][2];
            nuevas_posiciones_arr[3 * i] = 0.0f;
            nuevas_posiciones_arr[3 * i + 1] = 0.0f;
            nuevas_posiciones_arr[3 * i + 2] = 0.0f;
            velocidades_arr[3 * i] = cubeVelocities[i][0];
            velocidades_arr[3 * i + 1] = cubeVelocities[i][1];
            velocidades_arr[3 * i + 2] = cubeVelocities[i][2];
            nuevas_velocidades_arr[3 * i] = 0.0f;
            nuevas_velocidades_arr[3 * i + 1] = 0.0f;
            nuevas_velocidades_arr[3 * i + 2] = 0.0f;

        }
        
        cudaError_t cudaStatus = choques_cuda(posiciones_anteriores_arr, nuevas_posiciones_arr, velocidades_arr, nuevas_velocidades_arr);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addWithCuda failed!");
            return 1;
        }

        for (int i = 0; i < n_particulas; i++) {
            cubeVelocities[i][0] = nuevas_velocidades_arr[3*i];
            cubeVelocities[i][1] = nuevas_velocidades_arr[3*i + 1];
            cubeVelocities[i][2] = nuevas_velocidades_arr[3*i + 2];
            cubePositions[i][0] = nuevas_posiciones_arr[3 * i];
            cubePositions[i][1] = nuevas_posiciones_arr[3 * i + 1];
            cubePositions[i][2] = nuevas_posiciones_arr[3 * i + 2];
        }

        //calcular velocidades

        for (int i = 0; i < n_particulas; i++) {

            //rebotes con paredes
            if (abs(cubePositions[i][0]) + radio > 1.0) {
                cubePositions[i][0] = (cubePositions[i][0] / abs(cubePositions[i][0])) * (1 - radio);
                cubeVelocities[i][0] = -cubeVelocities[i][0];
            }

            if (abs(cubePositions[i][1]) + radio > 1.0) {
                cubePositions[i][1] = (cubePositions[i][1] / abs(cubePositions[i][1])) * (1 - radio);
                cubeVelocities[i][1] = -cubeVelocities[i][1];
            }

            if (abs(cubePositions[i][2]) + radio > 1.0) {
                cubePositions[i][2] = (cubePositions[i][2] / abs(cubePositions[i][2])) * (1 - radio);
                cubeVelocities[i][2] = -cubeVelocities[i][2];
            }

            //aplicar gravedad y roce
            cubeVelocities[i] = (cubeVelocities[i] + glm::vec3(0.0f, -gravedad * deltaTime, 0.0f)) * (1 - deltaTime * roce);

            /*
            //detener rebotes chicos
            if (glm::length(cubeVelocities[i]) < 0.01f && cubePositions[i][1] - radio <= -0.99) {
                cubeVelocities[i] = glm::vec3(0.0f, 0.0f, 0.0f);
                cubePositions[i][1] = -1.0f + radio;
            }
            */

            //actualizar posiciones y velocidades
            cubePositions[i] = cubePositions[i] + cubeVelocities[i] * deltaTime;
        }

        // renderear esferas

        for (unsigned int i = 0; i < n_particulas; i++)
        {
            // calculate the model matrix for each object and pass it to shader before drawing
            glm::mat4 model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
            model = glm::translate(model, cubePositions[i]);
            float angle = 20.0f * i;
            model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
            ourShader.setMat4("model", model);
            ourShader.setVec4("color", glm::vec4(abs(cos(i)), abs(sin(i)), abs(cos(i * i) + sin(i + i)) / 2, 1));

            //dibujar esfera
            sphere_draw();
        }

        intervalo_fps += deltaTime;
        count_fps += 1;

        if (intervalo_fps > periodo_fps) {
            fps = count_fps / intervalo_fps;
            intervalo_fps = 0.0f;
            count_fps = 0;
        }
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << fps;
        std::string s = stream.str();
        RenderText(shader, "FPS: " + s, 25.0f, 25.0f, 0.5f, glm::vec3(0.5, 0.8f, 0.2f));


        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }



    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}


// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}

void RenderText(Shader& shader, std::string text, float x, float y, float scale, glm::vec3 color)
{
    // activate corresponding render state	
    shader.use();
    glUniform3f(glGetUniformLocation(shader.ID, "textColor"), color.x, color.y, color.z);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(VAO_text);

    // iterate through all characters
    std::string::const_iterator c;
    for (c = text.begin(); c != text.end(); c++)
    {
        Character ch = Characters[*c];

        float xpos = x + ch.Bearing.x * scale;
        float ypos = y - (ch.Size.y - ch.Bearing.y) * scale;

        float w = ch.Size.x * scale;
        float h = ch.Size.y * scale;
        // update VBO for each character
        float vertices[6][4] = {
            { xpos,     ypos + h,   0.0f, 0.0f },
            { xpos,     ypos,       0.0f, 1.0f },
            { xpos + w, ypos,       1.0f, 1.0f },

            { xpos,     ypos + h,   0.0f, 0.0f },
            { xpos + w, ypos,       1.0f, 1.0f },
            { xpos + w, ypos + h,   1.0f, 0.0f }
        };
        // render glyph texture over quad
        glBindTexture(GL_TEXTURE_2D, ch.TextureID);
        // update content of VBO memory
        glBindBuffer(GL_ARRAY_BUFFER, VBO_text);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices); // be sure to use glBufferSubData and not glBufferData

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        // render quad
        glDrawArrays(GL_TRIANGLES, 0, 6);
        // now advance cursors for next glyph (note that advance is number of 1/64 pixels)
        x += (ch.Advance >> 6) * scale; // bitshift by 6 to get value in pixels (2^6 = 64 (divide amount of 1/64th pixels by 64 to get amount of pixels))
    }
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}




