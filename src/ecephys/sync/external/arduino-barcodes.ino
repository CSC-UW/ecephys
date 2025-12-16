/*
  32-bit digital barcodes for synchronizing data streams
*/

int OUTPUT_PIN = 13;
int BARCODE_BITS = 32;
int INTER_BARCODE_INTERVAL = 30; // s

long barcode;

void setup() {
  
  pinMode(OUTPUT_PIN, OUTPUT); // initialize digital pin

  randomSeed(analogRead(0));

  barcode = random(0, pow(2,BARCODE_BITS));
  
}

void loop() {

  barcode += 1; // increment barcode on each cycle

  digitalWrite(OUTPUT_PIN, HIGH);   // initialize with 20 ms pulse
  delay(20);
  digitalWrite(OUTPUT_PIN, LOW);    // set to low value for 20 ms
  delay(20);

  for (int i = 0; i < BARCODE_BITS; i++)
  {
    if ((barcode >> i) & 1)
      digitalWrite(OUTPUT_PIN,HIGH);
    else
      digitalWrite(OUTPUT_PIN,LOW);

     delay((INTER_BARCODE_INTERVAL - 1) * 32 / BARCODE_BITS);
  }
  
  digitalWrite(OUTPUT_PIN, LOW);         // write final low value
  delay(INTER_BARCODE_INTERVAL * 1000);  // wait for interval

}
