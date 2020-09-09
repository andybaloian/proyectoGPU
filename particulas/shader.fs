#version 330 core
out vec4 FragColor;
  
in vec3 ourColor;

uniform vec4 color;

void main()
{
    FragColor = color;
}