#include <LiquidCrystal_I2C.h>      //LiquidCrystal 라이브러리 추가
#include <string.h>
#include <stdlib.h>

LiquidCrystal_I2C lcd(0x3F, 16, 2); //lcd 객체 선언

int old_key1 = 0, new_key1 = 0;
int old_key2 = 0, new_key2 = 0;
int old_key3 = 0, new_key3 = 0;

#define LIMIT_PIN 9
#define SOLENOID_PIN 8
#define X_STEP_PIN 7
#define X_DIR_PIN 6
#define Y_STEP_PIN1 5
#define Y_DIR_PIN1 4
#define Y_STEP_PIN2 3
#define Y_DIR_PIN2 2

bool SOL_init = 0;

const int MOTOR_DELAY = 1000;

const int maxChar = 34;
char charVal[maxChar];
int charIter = 0;

char line1[16];
char line2[16];
bool lcd_flag = false;

void setup()
{
    Serial.begin(9600);
    pinMode(2, OUTPUT);
    pinMode(3, OUTPUT);
    pinMode(4, OUTPUT);
    pinMode(5, OUTPUT);
    pinMode(6, OUTPUT);
    pinMode(7, OUTPUT);
    pinMode(8, OUTPUT);
    pinMode(9, INPUT);
    lcd.begin();
    lcd.print("INITIALIZING...");
}

void loop()
{
    if(Serial.available() > 0)
    {
        char a = Serial.read();
        if(a == '`')
        {
            processSerial();
            Serial.println(charVal);

            //initializing part
            for(int i = 0; i < maxChar; ++i)
            {
                charVal[i] = '\0';
            }
            charIter = 0;
        }
        else
        {
            if(charIter < maxChar - 1)
            {
                charVal[charIter] = a;
                charIter += 1;
            }
        }
    }
}

int pick_integer(char* str) 
{ /* 문자열에서 숫자만 추출하는 함수 */
	int num=0, plus=0;
	
	for(int i=0; i < maxChar; i++)
    { //문자열의 길이만큼 반복
        if(str[i] > 47 && str[i] < 58) 
        {num = num*10 + str[i]-48;}		
	}
    return num;
}

void processSerial()
{
    char decision = charVal[0];
    int tmpInt = pick_integer(charVal);

    if(decision == 'p')
    {
        digitalWrite(SOLENOID_PIN, HIGH);
        delay(200);
    }
    else if(decision == 'P')
    {
        digitalWrite(SOLENOID_PIN, LOW);
        delay(70);
    }
    else if(decision == 'd')
    {
        ctrl_Y_motor_d(Y_DIR_PIN1, Y_DIR_PIN2, Y_STEP_PIN1, Y_STEP_PIN2, tmpInt);
    }
    else if(decision == 'u')
    {
        ctrl_Y_motor_u(Y_DIR_PIN1, Y_DIR_PIN2, Y_STEP_PIN1, Y_STEP_PIN2, tmpInt);
    }
    else if(decision == 'r')
    {
        ctrl_motor(X_DIR_PIN, X_STEP_PIN, HIGH, tmpInt);
    }
    else if(decision == 'l')
    {
         ctrl_motor(X_DIR_PIN, X_STEP_PIN, LOW, tmpInt);
    }
    else if(decision == 'i')
    {
        SOL_init = 0;
        int limit_step = act_limit();
        char limit_char[4];
        itoa(limit_step, limit_char, 10);
        strcpy(charVal, limit_char);
    }
    else if(decision == '+')
    {
        rotate_cw(Y_DIR_PIN1, Y_DIR_PIN2, Y_STEP_PIN1, Y_STEP_PIN2, tmpInt);
    }
    else if(decision == '-')
    {
        rotate_ccw(Y_DIR_PIN1, Y_DIR_PIN2, Y_STEP_PIN1, Y_STEP_PIN2, tmpInt);
    }
    else if(decision == 'f')
    {
        go_forward(Y_DIR_PIN1, Y_DIR_PIN2, Y_STEP_PIN1, Y_STEP_PIN2, 5);
    }

    else if(decision == '!')
    {
        int line_flag = 0;
        int line_iter;
        for(int i = 0; i < 16; ++i)
        {
            line1[i] = '\0';
            line2[i] = '\0';
        }

        for(int i = 0; i <= charIter; ++i)
        {
            if (charVal[i] == '!')
            {
                line_flag += 1;
                line_iter = 0;
            }
            else if (line_flag == 1)
            {
                if(line_iter < 16)
                {
                    line1[line_iter] = charVal[i];
                    line_iter += 1;
                }
            }
            else if (line_flag == 2)
            {   
                if(line_iter < 16)
                {
                    line2[line_iter] = charVal[i];
                    line_iter += 1;
                }
            }
        }
        strcpy(charVal, "pr");
        lcd_flag = true;
    }

    if(lcd_flag)
    {
        lcd.clear();
        lcd.print(line1);
        lcd.setCursor(0, 1);
        lcd.print(line2);
        lcd_flag = false;
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
    for(int i = 0; i < motor_step * 4; ++i)
    {
        digitalWrite(motor_step_pin, HIGH);
        delayMicroseconds(MOTOR_DELAY);
        digitalWrite(motor_step_pin, LOW);
        delayMicroseconds(MOTOR_DELAY);
    }
}

void ctrl_Y_motor_d(int motor_dir_pin1, int motor_dir_pin2, int motor_step_pin1, int motor_step_pin2, int motor_step)
{
    digitalWrite(motor_dir_pin1, HIGH);
    digitalWrite(motor_dir_pin2, LOW);
    for(int i = 0; i < motor_step * 2; ++i)
    {
        digitalWrite(motor_step_pin1, HIGH);
        digitalWrite(motor_step_pin2, HIGH);
        delayMicroseconds(MOTOR_DELAY * 2);
        digitalWrite(motor_step_pin1, LOW);
        digitalWrite(motor_step_pin2, LOW);
        delayMicroseconds(MOTOR_DELAY * 2);
    }
}

void ctrl_Y_motor_u(int motor_dir_pin1, int motor_dir_pin2, int motor_step_pin1, int motor_step_pin2, int motor_step)
{
    digitalWrite(motor_dir_pin1, LOW);
    digitalWrite(motor_dir_pin2, HIGH);
    for(int i = 0; i < motor_step * 2; ++i)
    {
        digitalWrite(motor_step_pin1, HIGH);
        digitalWrite(motor_step_pin2, HIGH);
        delayMicroseconds(MOTOR_DELAY * 2);
        digitalWrite(motor_step_pin1, LOW);
        digitalWrite(motor_step_pin2, LOW);
        delayMicroseconds(MOTOR_DELAY * 2);
    }
}

int act_limit()
{
    int ans = 0;
    while (SOL_init == 0)
    {
        bool limit_data = digitalRead(LIMIT_PIN);
        if (limit_data == 0)
        {
            ans += 1;
            ctrl_motor(X_DIR_PIN, X_STEP_PIN, LOW, 1);
        }
        else
        {
            SOL_init = 1;
        }
    }
    return ans;
}

void rotate_cw(int motor_dir_pin1, int motor_dir_pin2, int motor_step_pin1, int motor_step_pin2, int motor_step)
{
    digitalWrite(motor_dir_pin1, HIGH);
    //digitalWrite(motor_dir_pin2, HIGH);
    //i < motor_step * n
    for(int i = 0; i < motor_step * 5; ++i)
    {
        digitalWrite(motor_step_pin1, HIGH);
        //digitalWrite(motor_step_pin2, HIGH);
        delayMicroseconds(MOTOR_DELAY * 5);
        digitalWrite(motor_step_pin1, LOW);
        //digitalWrite(motor_step_pin2, LOW);
        delayMicroseconds(MOTOR_DELAY * 5);
    }
}

void rotate_ccw(int motor_dir_pin1, int motor_dir_pin2, int motor_step_pin1, int motor_step_pin2, int motor_step)
{
    //digitalWrite(motor_dir_pin1, LOW);
    digitalWrite(motor_dir_pin2, LOW);
    //i < motor_step * n
    for(int i = 0; i < motor_step * 5; ++i)
    {
        //digitalWrite(motor_step_pin1, HIGH);
        digitalWrite(motor_step_pin2, HIGH);
        delayMicroseconds(MOTOR_DELAY * 5);
        //digitalWrite(motor_step_pin1, LOW);
        digitalWrite(motor_step_pin2, LOW);
        delayMicroseconds(MOTOR_DELAY * 5);
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
        delayMicroseconds(MOTOR_DELAY * 5);
        digitalWrite(motor_step_pin1, LOW);
        digitalWrite(motor_step_pin2, LOW);
        delayMicroseconds(MOTOR_DELAY * 5);
    }
}