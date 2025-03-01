int position_pin[] = {1,2,3,4};               //4자리 결정 핀
int segment_pin[] = {5,6,7,8,9,10,11,12};     //세그먼트 제어 핀
const int delayTime = 5;                      //일시정지 시간
 
//0 ~ 9를 표현하는 세그먼트 값
byte data[] = {0xFC, 0x60, 0xDA, 0xF2, 0x66, 0xB6, 0xBE, 0xE4, 0xFE, 0xE6};
 
void setup() {
  //4자리 결정 핀 출력용으로 설정
  for(int i = 0; i < 4; i++) {
     pinMode(position_pin[i], OUTPUT);
  }
 
  //세그먼트 제어 핀 출력용으로 설정
  for(int i = 0; i < 8; i++) {
    pinMode(segment_pin[i], OUTPUT);
  }
}
 
void loop() {
  show(2,2);                //두 번째 자리, 2출력
  delay(delayTime);         //0.005초 일시정지
  show(3,3);                //세 번째 자리, 3출력
  delay(delayTime);         //0.005초 일시정지

  count();                //count함수 호출
}
 
void show(int position, int number) {
  //4자리 중 원하는 자리 선택
  for(int i = 0; i < 4; i++) {
    if(i + 1 == position){
      digitalWrite(position_pin[i], HIGH);
    } else {
      digitalWrite(position_pin[i], LOW);
    }
  }
  
  //8개 세그먼트를 제어해서 원하는 숫자 출력
  for(int i = 0; i < 8; i++){
     byte segment = (data[number] & (0x01 << i)) >> i;
     if(segment == 1){
       digitalWrite(segment_pin[7 - i], LOW);
     } else {
       digitalWrite(segment_pin[7 - i], HIGH);
     }
  }
}
 

void count() {
  for(int k=0; k<5000; k++){
    for(int i = 0; i < 10; i++) {
      for(int j = 0; j < 10; j++) {
        show(2,i);
        delay(delayTime);
        show(3,j);
        delay(delayTime);
      }
    }
    delay(delayTime);
  }
}
