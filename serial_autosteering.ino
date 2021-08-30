#include <ctype.h>
#include <stdlib.h>
#define Y_DIR_PIN1 5
#define Y_STEP_PIN1 4
#define Y_DIR_PIN2 3
#define Y_STEP_PIN2 2

#define MAX_REPOS 7

char num_repos[MAX_REPOS];
int num_repos_iter = 0;
char dir_flag = '\0';

const int MOTOR_DELAY = 500;

String ser_data = "";

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
    while(Serial.available() > 0)
    {
        char rec = Serial.read();
        ser_data += rec;
        
        //실행 영역

        if (rec == '+')
        {
            Serial.println(ser_data);
            dir_flag = 'r';
        }
        else if (rec == '-')
        {
            Serial.println(ser_data);
            dir_flag = 'l';
        }
        else if (isdigit(rec) != 0)
        {
            Serial.println(ser_data);
            num_repos[num_repos_iter] = rec;
            num_repos_iter += 1;
        }
        else if (rec == '`')
        {
            int tmp = atoi(num_repos);
            
            if (dir_flag == 'r')
            {
                rotate_to_right(Y_DIR_PIN1, Y_DIR_PIN2, Y_STEP_PIN1, Y_STEP_PIN2, tmp);
            }
            else if (dir_flag == 'l')
            {
                rotate_to_left(Y_DIR_PIN1, Y_DIR_PIN2, Y_STEP_PIN1, Y_STEP_PIN2, tmp);
            }

            Serial.println(tmp);
            
            for(int i = 0; i < MAX_REPOS; ++i)
            {
                num_repos[i] = '\0';
            }
            dir_flag = '\0';
            num_repos_iter = 0;
        }
        else
        {
            Serial.println("U");
        }
        ser_data = "";
        delay(1);
    }
}

void rotate_to_left(int motor_dir_pin1, int motor_dir_pin2, int motor_step_pin1, int motor_step_pin2, int motor_step)
{
    digitalWrite(motor_dir_pin1, HIGH);
    digitalWrite(motor_dir_pin2, HIGH);
    //i < motor_step * n
    for(int i = 0; i < motor_step * 5; ++i)
    {
        digitalWrite(motor_step_pin1, HIGH);
        //digitalWrite(motor_step_pin2, HIGH);
        delayMicroseconds(MOTOR_DELAY * 2);
        digitalWrite(motor_step_pin1, LOW);
        //digitalWrite(motor_step_pin2, LOW);
        delayMicroseconds(MOTOR_DELAY * 2);
    }
}

void rotate_to_right(int motor_dir_pin1, int motor_dir_pin2, int motor_step_pin1, int motor_step_pin2, int motor_step)
{
    digitalWrite(motor_dir_pin1, LOW);
    digitalWrite(motor_dir_pin2, LOW);
    //i < motor_step * n
    for(int i = 0; i < motor_step * 5; ++i)
    {
        //digitalWrite(motor_step_pin1, HIGH);
        digitalWrite(motor_step_pin2, HIGH);
        delayMicroseconds(MOTOR_DELAY * 2);
        //digitalWrite(motor_step_pin1, LOW);
        digitalWrite(motor_step_pin2, LOW);
        delayMicroseconds(MOTOR_DELAY * 2);
    }
}