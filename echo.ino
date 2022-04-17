void setup(){
    Serial.begin(9600);
    char data;

}

void loop(){
    if(Serial.available()){
       data =  Serial.read();
       Serial.println(data);
    }
}