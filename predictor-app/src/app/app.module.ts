import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';             // do ngModel
import { HttpClientModule } from '@angular/common/http';  // do HTTP

import { AppComponent } from './app.component';
import { PredictComponent } from './predict.component';  // bo oba pliki są w tym samym folderze

@NgModule({
  declarations: [
    AppComponent,
    PredictComponent
  ],
  imports: [
    BrowserModule,
    FormsModule,
    HttpClientModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }