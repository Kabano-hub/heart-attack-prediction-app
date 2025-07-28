import { Component } from '@angular/core';
import { PredictionService } from './prediction.service';

@Component({
  selector: 'app-predict',
  templateUrl: './predict.component.html'
})
export class PredictComponent {
inputData: any = {
  age: 50,
  sex: 1,
  cp: 1,
  trestbps: 120,
  chol: 240,
  fbs: 0,
  restecg: 1,
  thalach: 150,
  exang: 0,
  oldpeak: 1.0,
  slope: 2,
  ca: 0,
  thal: 3,
};

  result: any = null;

  constructor(private predictionService: PredictionService) {}

  predict() {
    this.predictionService.predict(this.inputData).subscribe({
      next: res => this.result = res,
      error: err => console.error(err)
    });
  }
}
