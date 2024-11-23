// Include necessary headers
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <cmath>

// Vertex Shader Source Code
const char* vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
)";

// Fragment Shader Source Code
const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform vec2 iResolution;
uniform float iTime;
uniform sampler3D noiseTexture;

void main() {
    vec2 uv = TexCoord * iResolution / max(iResolution.x, iResolution.y);
    uv = uv * 2.0 - 1.0; // Center UV coordinates

    // Create a vertical gradient for fire fade-out and smoke
    float gradient = 1.0 - abs(uv.y);

    // Sample 3D noise texture with a time-based Z offset to animate
    float noiseValue = texture(noiseTexture, vec3(uv * 0.5 + 0.5, iTime * 0.1)).r;
    noiseValue = pow(noiseValue, 2.0); // Adjust contrast for vivid effect

    // Fire colors: bright yellow-orange to deep orange
    vec3 fireColorBright = vec3(1.0, 0.7, 0.0);
    vec3 fireColorDeep = vec3(1.0, 0.3, 0.0);

    // Smoke colors: dark grey to light grey
    vec3 smokeColorDark = vec3(0.2, 0.2, 0.2);
    vec3 smokeColorLight = vec3(0.7, 0.7, 0.7);

    // Calculate intensity for both fire and smoke
    float fireIntensity = smoothstep(0.2, 1.0, gradient * noiseValue);
    float smokeIntensity = smoothstep(0.0, 0.5, gradient * noiseValue); // Smoke fades out gradually

    // Blend fire and smoke colors based on intensity
    vec3 fireColor = mix(fireColorDeep, fireColorBright, fireIntensity);
    vec3 smokeColor = mix(smokeColorDark, smokeColorLight, smokeIntensity);

    // Mix smoke and fire based on vertical position and intensity
    vec3 finalColor = mix(smokeColor, fireColor, smoothstep(-0.1, 0.3, uv.y + 0.5 * fireIntensity));

    // Adjust alpha based on fire and smoke intensity for smooth blending
    float alpha = max(fireIntensity, smokeIntensity) * 0.7; // Overall alpha for blending
    FragColor = vec4(finalColor, alpha);
}
)";


// Function to compile shaders and check for errors
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    // Error handling
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        const char* shaderType = (type == GL_VERTEX_SHADER) ? "VERTEX" : "FRAGMENT";
        std::cerr << "ERROR::SHADER::" << shaderType << "::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    return shader;
}

// Permutation table for Perlin noise
int p[512];

void initPermutation() {
    int permutation[] = { 151,160,137,91,90,15,
       131,13,201,95,96,53,194,233,7,225,
       140,36,103,30,69,142,8,99,37,240,
       21,10,23,190,6,148,247,120,234,75,
       0,26,197,62,94,252,219,203,117,35,
       11,32,57,177,33,88,237,149,56,87,
       174,20,125,136,171,168,68,175,74,
       165,71,134,139,48,27,166,77,146,158,
       231,83,111,229,122,60,211,133,230,220,
       105,92,41,55,46,245,40,244,102,143,
       54,65,25,63,161,1,216,80,73,209,
       76,132,187,208,89,18,169,200,196,
       135,130,116,188,159,86,164,100,109,
       198,173,186,3,64,52,217,226,250,
       124,123,5,202,38,147,118,126,255,
       82,85,212,207,206,59,227,47,16,58,
       17,182,189,28,42,223,183,170,213,
       119,248,152,2,44,154,163,70,221,
       153,101,155,167,43,172,9,129,22,
       39,253,19,98,108,110,79,113,224,
       232,178,185,112,104,218,246,97,228,
       251,34,242,193,238,210,144,12,191,
       179,162,241,81,51,145,235,249,14,
       239,107,49,192,214,31,181,199,106,
       157,184,84,204,176,115,121,50,45,
       127,4,150,254,138,236,205,93,222,
       114,67,29,24,72,243,141,128,195,
       78,66,215,61,156,180
    };
    for (int i = 0; i < 256; i++) {
        p[i] = permutation[i];
        p[256 + i] = permutation[i];
    }
}


// Fade function for Perlin noise
float fade(float t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

// Linear interpolation
float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// Gradient function
float grad(int hash, float x, float y, float z) {
    int h = hash & 15;
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : (h == 12 || h == 14) ? x : z;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

// Perlin noise function in 3D
float noise(float x, float y, float z) {
    int X = (int)floor(x) & 255;
    int Y = (int)floor(y) & 255;
    int Z = (int)floor(z) & 255;

    x -= floor(x);
    y -= floor(y);
    z -= floor(z);

    float u = fade(x);
    float v = fade(y);
    float w = fade(z);

    int A = p[X] + Y, AA = p[A] + Z, AB = p[A + 1] + Z;
    int B = p[X + 1] + Y, BA = p[B] + Z, BB = p[B + 1] + Z;

    float res = lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z),
        grad(p[BA], x - 1, y, z)),
        lerp(u, grad(p[AB], x, y - 1, z),
            grad(p[BB], x - 1, y - 1, z))),
        lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1),
            grad(p[BA + 1], x - 1, y, z - 1)),
            lerp(u, grad(p[AB + 1], x, y - 1, z - 1),
                grad(p[BB + 1], x - 1, y - 1, z - 1))));

    return (res + 1.0) / 2.0; // Normalize result to [0,1]
}

// Fractal Brownian Motion (fBM) function
float fbm(float x, float y, float z) {
    float total = 0.0;
    float frequency = 1.0;
    float amplitude = 1.0;
    float maxValue = 0.0;
    const int octaves = 10; // Increased number of octaves for finer detail
    const float lacunarity = 2.0;
    const float gain = 0.5;

    for (int i = 0; i < octaves; i++) {
        total += noise(x * frequency, y * frequency, z * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    return total / maxValue;
}

// Function to create a 3D Perlin noise texture
GLuint createPerlinNoiseTexture(int size) {
    std::vector<float> data(size * size * size);

    initPermutation(); // Initialize the permutation table

    for (int x = 0; x < size; ++x) {
        for (int y = 0; y < size; ++y) {
            for (int z = 0; z < size; ++z) {
                float fx = (float)x / (float)size;
                float fy = (float)y / (float)size;
                float fz = (float)z / (float)size;

                // Use fbm for richer noise
                float value = fbm(fx * 15.0f, fy * 15.0f, fz * 15.0f); // Increased frequency for finer details
                data[x + size * (y + size * z)] = value;
            }
        }
    }

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_3D, textureID);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, size, size, size, 0, GL_RED, GL_FLOAT, data.data());

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_MIRRORED_REPEAT);

    return textureID;
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create a windowed mode window and its OpenGL context
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "Animated Fire and Smoke", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }


    // Build and compile shaders
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    // Shader program
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check for linking errors
    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    // Delete shaders; they're linked into our program now and no longer necessary
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Set up vertex data (and buffer(s)) and configure vertex attributes
    float vertices[] = {
        // positions        // texture coords
        -1.0f, -1.0f, 0.0f,  0.0f, 0.0f, // Bottom-left
         1.0f, -1.0f, 0.0f,  1.0f, 0.0f, // Bottom-right
         1.0f,  1.0f, 0.0f,  1.0f, 1.0f, // Top-right
        -1.0f,  1.0f, 0.0f,  0.0f, 1.0f  // Top-left
    };
    unsigned int indices[] = {
        0, 1, 2, // First triangle
        2, 3, 0  // Second triangle
    };

    GLuint VBO, VAO, EBO;

    // Generate buffers and arrays
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // Bind and set vertex buffers and configure vertex attributes
    glBindVertexArray(VAO);

    // Load data into vertex buffers
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Load data into element buffers
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture Coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Create Perlin noise texture
    GLuint noiseTexture = createPerlinNoiseTexture(64);

    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Get window dimensions
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    // Main render loop
    while (!glfwWindowShouldClose(window)) {
        // Input handling
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        // Render commands here
        glViewport(0, 0, width, height);
        glClearColor(0.0f, 0.0f, 0.0f, 1); // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT);

        // Activate shader program
        glUseProgram(shaderProgram);

        // Set the shader uniforms
        glm::mat4 model = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)); // Rotate the screen by 180 degrees
        glm::mat4 view = glm::mat4(1.0f);  // Identity matrix for view
        glm::mat4 projection = glm::ortho(-1.0f, 1.0f, -1.0f * (float)height / (float)width, 1.0f * (float)height / (float)width, -1.0f, 1.0f); // Orthographic projection

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        glUniform2f(glGetUniformLocation(shaderProgram, "iResolution"), (float)width, (float)height);
        glUniform1f(glGetUniformLocation(shaderProgram, "iTime"), (float)glfwGetTime());


        // Bind 3D noise texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, noiseTexture);
        glUniform1i(glGetUniformLocation(shaderProgram, "noiseTexture"), 0);

        // Render the quad
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // Swap front and back buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();

        // Update window size in case of resize
        glfwGetFramebufferSize(window, &width, &height);
    }

    // Deallocate resources
    glDeleteTextures(1, &noiseTexture);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);

    // Terminate GLFW
    glfwTerminate();
    return 0;
}