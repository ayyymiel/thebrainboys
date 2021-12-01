#include <Servo.h>
#include <SoftwareSerial.h>
SoftwareSerial HC06(10, 11); //HC06-TX Pin 5, HC06-RX to Arduino Pin 4

Servo frh;
Servo frl;

Servo flh;
Servo fll;

Servo brh;
Servo brl;

Servo blh;
Servo bll;

//int home_pos = 85;
int i=0;

int frh_h = 105;
int frl_h = 65;
int flh_h = 68;
int fll_h = 105;
int brh_h = 75;
int brl_h = 120;
int blh_h = 98;
int bll_h = 75;

void setup() {
  // put your setup code here, to run once:
HC06.begin(9600); //Baudrate 9600 , Choose your own baudrate

frh.attach(8);
frh.write(frh_h);
frl.attach(9);
frl.write(frl_h+90);

flh.attach(2);
flh.write(flh_h);
fll.attach(3);
fll.write(fll_h-90);

brh.attach(6);
brh.write(brh_h);
brl.attach(7);
brl.write(brl_h-90);

blh.attach(4);
blh.write(blh_h);
bll.attach(5);
bll.write(bll_h+90);

stand();
}

void loop() {

  // put your main code here, to run repeatedly:
  if(HC06.available() > 0){
    char receive = HC06.read(); //Read from Serial Communication
    //If received data is 1, turn on the LED and send back the sensor data
    if(receive == 'f') {
      forward();
      HC06.println("forward");
    }
    else if (receive =='b'){
      backward();
      HC06.println("backward");
    }
    else if (receive =='r'){
      right();
      HC06.println("right");
    }
    else if (receive =='l'){
      left();
      HC06.println("left");
    }
    else {
      layout();
      HC06.println("other");
    }
  }

}

void stand(){
//hip movements

frh.write(frh_h+45);
flh.write(flh_h-45);
brh.write(brh_h-45);
blh.write(blh_h+45);
delay(150);
//leg movements
frl.write(frl_h+90);
fll.write(fll_h-90);
brl.write(brl_h-90);
bll.write(bll_h+90);
}

void layout(){
frh.write(frh_h);
frl.write(frl_h-30);

flh.write(flh_h);
fll.write(fll_h+40);

brh.write(brh_h);
brl.write(brl_h+30);

blh.write(blh_h);
bll.write(bll_h);

delay(1000);
stand();
delay(500);

frh.write(frh_h);
frl.write(frl_h-30);

flh.write(flh_h);
fll.write(fll_h+40);

brh.write(brh_h);
brl.write(brl_h+30);

blh.write(blh_h);
bll.write(bll_h);
delay(1000);
stand();
}


//walk forward
void forward(){

  delay(150);
  stand();
  for (i=0; i<=3;i++){
    frh.write(frh_h+60);
    frl.write(frl_h+40);
    delay(50);
    blh.write(blh_h-60);
    bll.write(bll_h+40);
    delay(50);

    frl.write(frl_h+90);
    bll.write(bll_h+90);
    delay(50);

    frh.write(frh_h);
    blh.write(blh_h);
    delay(50);


    flh.write(flh_h-60);
    fll.write(fll_h-40);
    delay(50);
    brh.write(brh_h+60);
    brl.write(brl_h-40);
    delay(50);

    fll.write(fll_h-90);
    brl.write(brl_h-90);
    delay(50);

    flh.write(flh_h);
    brh.write(brh_h);
    delay(50);
  }

}

//walk backward
void backward(){
  delay(150);
  stand();
  for (i=0; i<=3;i++){
    blh.write(blh_h+60);
    bll.write(bll_h+40);
    delay(100);
    frh.write(frh_h-60);
    frl.write(frl_h+40);
    delay(50);

    bll.write(bll_h+80);
    frl.write(frl_h+90);
    delay(50);

    blh.write(blh_h);
    frh.write(frh_h);
    delay(50);

    brh.write(brh_h-60);
    brl.write(brl_h-40);
    delay(100);
    flh.write(flh_h+60);
    fll.write(fll_h-40);
    delay(50);

    brl.write(brl_h-80);
    fll.write(fll_h-90);
    delay(50);

    brh.write(brh_h);
    flh.write(flh_h);
    delay(50);

  }

}

//right turn
void right(){
  delay(200);
  stand();

    for (i=0; i<=2; i++){

      //front right leg
      frl.write(frl_h+30);
      delay(50);
      brh.write(brh_h+35);
      //delay(50);
      flh.write(flh_h-65);
      delay(50);
      frh.write(frh_h-55);
      delay(50);
      frl.write(frl_h+90);
      delay(50);

      //back right leg
      brl.write(brl_h-30);
      delay(50);
      blh.write(blh_h+35);
      //delay(50);
      frh.write(frh_h+40);
      delay(50);
      brh.write(brh_h-55);
      delay(50);
      brl.write(brl_h-90);
      delay(50);

      //back left leg
      bll.write(bll_h+30);
      delay(50);
      flh.write(flh_h+35);
      //delay(50);
      brh.write(brh_h-40);
      delay(50);
      blh.write(brh_h-55);
      delay(50);
      bll.write(bll_h+90);
      delay(50);

      //front left leg
      fll.write(fll_h-30);
      delay(50);
      frh.write(frh_h+35);
      //delay(50);
      blh.write(blh_h+35);
      delay(50);
      flh.write(flh_h-55);
      delay(50);
      fll.write(fll_h-90);
      delay(50);
    }
}

////// need to change//
//left turn
void left(){
  delay(200);
  stand();

    for (i=0; i<=2; i++){

      //front left leg
      fll.write(fll_h-30);
      delay(50);
      blh.write(blh_h-35);
      //delay(50);
      frh.write(frh_h+40);
      delay(50);
      flh.write(flh_h+55);
      delay(50);
      fll.write(fll_h-90);
      delay(50);

      //back left leg
      bll.write(bll_h+30);
      delay(50);
      brh.write(brh_h-35);
      //delay(50);
      flh.write(flh_h-40);
      delay(50);
      blh.write(blh_h+55);
      delay(50);
      bll.write(bll_h+90);
      delay(50);

      //back right leg
      brl.write(brl_h-30);
      delay(50);
      frh.write(frh_h-35);
      //delay(50);
      blh.write(blh_h+40);
      delay(50);
      brh.write(flh_h+55);
      delay(50);
      brl.write(fll_h-90);
      delay(50);

      //front right leg
      frl.write(frl_h+30);
      delay(50);
      flh.write(brh_h-35);
      //delay(50);
      brh.write(flh_h-40);
      delay(50);
      frh.write(frh_h+55);

      delay(50);
      frl.write(frl_h+90);
      delay(50);
    }
}