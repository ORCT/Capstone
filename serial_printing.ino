#include<ctype.h>
#include<stdlib.h>
#define SOLENOID_PIN 2
#define X_DIR_PIN 3
#define X_STEP_PIN 4
#define Y_DIR_PIN 5
#define Y_STEP_PIN 6
#define RIGHT OUTPUT  // Motor
#define LEFT LOW

#define MAX_REPOS 7

const int MOTOR_DELAY = 1500;
const int SERIAL_TIME_DELAY = 10;

String ser_data;
char num_repos[MAX_REPOS];
int num_repos_iter = 0;
char dir_flag;

void setup()
{
    pinMode(2, OUTPUT);
    pinMode(3, OUTPUT);
    pinMode(4, OUTPUT);
    pinMode(5, OUTPUT);
    pinMode(6, OUTPUT);
    pinMode(7, OUTPUT);
    Serial.begin(9600);
}

void loop()
{
    while(Serial.available() > 0)
    {
        char rec = Serial.read();
        ser_data += rec;
        
        //실행 영역

        if (rec == 'p')
        {
            Serial.println(ser_data);
            digitalWrite(SOLENOID_PIN, HIGH);
            delay(700);
        }
        else if (rec == 'P')
        {
            Serial.println(ser_data);
            digitalWrite(SOLENOID_PIN, LOW);
            delay(700);
        }
        else if (rec == 'r')
        {
            Serial.println(ser_data);
            dir_flag = 'r';
        }
        else if (rec == 'l')
        {
            Serial.println(ser_data);
            dir_flag = 'l';
        }
        else if (rec == 'd')
        {
            Serial.println(ser_data);
            dir_flag = 'd';
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
            if(dir_flag == 'r')
            {
                ctrl_motor(X_DIR_PIN, X_STEP_PIN, HIGH, tmp);
            }
            else if(dir_flag == 'l')
            {
                ctrl_motor(X_DIR_PIN, X_STEP_PIN, LOW, tmp);
            }
            else if(dir_flag == 'd')
            {
                ctrl_motor(Y_DIR_PIN, Y_STEP_PIN, HIGH, tmp);
            }

            Serial.println(tmp);
            if(dir_flag != '\0')
            {
                for(int i = 0; i < MAX_REPOS; ++i)
                {
                    num_repos[i] = '\0';
                }
                num_repos_iter = 0;
            }

            dir_flag = '\0';
        }
        else
        {
            Serial.println('U');
        }

        ser_data = "";
        delay(SERIAL_TIME_DELAY);
    }
}

void ctrl_motor(int motor_dir_pin, int motor_step_pin, int motor_dir, int motor_step)
{
    if(motor_dir == HIGH)
    {
        digitalWrite(motor_dir_pin, HIGH);
    }
    else
    {
        digitalWrite(motor_dir_pin, LOW);
    }
    for(int i = 0; i < motor_step * 2; ++i)
    {
        digitalWrite(motor_step_pin, HIGH);
        delayMicroseconds(MOTOR_DELAY);
        digitalWrite(motor_step_pin, LOW);
        delayMicroseconds(MOTOR_DELAY);
    }
}