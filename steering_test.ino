#include <ctype.h>
#include <stdlib.h>
#define Y_DIR_PIN1 5
#define Y_STEP_PIN1 4
#define Y_DIR_PIN2 3
#define Y_STEP_PIN2 2

#define MAX_REPOS 7

char num_repos[MAX_REPOS];
int num_repos_iter = 0;

const int MOTOR_DELAY = 1500;

void setup()
{
    pinMode(2, OUTPUT);
    pinMode(3, OUTPUT);
    pinMode(4, OUTPUT);
    pinMode(5, OUTPUT);
    Serial.begin(9600);
}

void loop()
{
    rotate_cw(Y_DIR_PIN1, Y_DIR_PIN2, Y_STEP_PIN1, Y_STEP_PIN2, 200);
    delayMicroseconds(MOTOR_DELAY * 2);
    go_forward(Y_DIR_PIN1, Y_DIR_PIN2, Y_STEP_PIN1, Y_STEP_PIN2, 200);
    delayMicroseconds(MOTOR_DELAY * 2);
    rotate_ccw(Y_DIR_PIN1, Y_DIR_PIN2, Y_STEP_PIN1, Y_STEP_PIN2, 200);

}

void rotate_cw(int motor_dir_pin1, int motor_dir_pin2, int motor_step_pin1, int motor_step_pin2, int motor_step)
{
    digitalWrite(motor_dir_pin1, HIGH);
    digitalWrite(motor_dir_pin2, HIGH);
    //i < motor_step * n
    for(int i = 0; i < motor_step * 5; ++i)
    {
        digitalWrite(motor_step_pin1, HIGH);
        digitalWrite(motor_step_pin2, HIGH);
        delayMicroseconds(MOTOR_DELAY * 2);
        digitalWrite(motor_step_pin1, LOW);
        digitalWrite(motor_step_pin2, LOW);
        delayMicroseconds(MOTOR_DELAY * 2);
    }
}

void rotate_ccw(int motor_dir_pin1, int motor_dir_pin2, int motor_step_pin1, int motor_step_pin2, int motor_step)
{
    digitalWrite(motor_dir_pin1, LOW);
    digitalWrite(motor_dir_pin2, LOW);
    //i < motor_step * n
    for(int i = 0; i < motor_step * 5; ++i)
    {
        digitalWrite(motor_step_pin1, HIGH);
        digitalWrite(motor_step_pin2, HIGH);
        delayMicroseconds(MOTOR_DELAY * 2);
        digitalWrite(motor_step_pin1, LOW);
        digitalWrite(motor_step_pin2, LOW);
        delayMicroseconds(MOTOR_DELAY * 2);
    }
}

void go_forward(int motor_dir_pin1, int motor_dir_pin2, int motor_step_pin1, int motor_step_pin2, int motor_step)
{
    digitalWrite(motor_dir_pin1, HIGH);
    digitalWrite(motor_dir_pin2, LOW);
    //i < motor_step * n
    for(int i = 0; i < motor_step * 5; ++i)
    {
        digitalWrite(motor_step_pin1, HIGH);
        digitalWrite(motor_step_pin2, HIGH);
        delayMicroseconds(MOTOR_DELAY * 2);
        digitalWrite(motor_step_pin1, LOW);
        digitalWrite(motor_step_pin2, LOW);
        delayMicroseconds(MOTOR_DELAY * 2);
    }
}