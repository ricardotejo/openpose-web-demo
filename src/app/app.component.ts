import { Component, ViewChild, OnInit, AfterViewInit, ElementRef } from '@angular/core';
import { loadFrozenModel, FrozenModel } from '@tensorflow/tfjs-converter';
import * as tfc from '@tensorflow/tfjs-core';
import { NamedTensorMap } from '@tensorflow/tfjs-core/dist/types';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html'
})
export class AppComponent implements AfterViewInit {
  title = 'app';
  model: FrozenModel;
  logdata = '';

  @ViewChild('inputImage', { static: true })
  inputImage: ElementRef;

  ngAfterViewInit() {
    this.load().then(() => {
      this.logdata += 'Loaded!\n';
    });
  }

  async load() {
    const MODEL_BASE_URL = 'assets/web_model';
    const MODEL_URL = MODEL_BASE_URL + '/tensorflowjs_model.pb';
    const WEIGHTS_URL = MODEL_BASE_URL + '/weights_manifest.json';

    this.model = await loadFrozenModel(MODEL_URL, WEIGHTS_URL);

  }

  execute() {
    this.run(this.inputImage.nativeElement).then(() => {
      // this.logdata += '';
    });

  }

  async run(imgData: HTMLImageElement) {
    let result: NamedTensorMap;
    const d0 = await tfc.time(() => {
      const inputTensor = tfc.fromPixels(imgData);
      const reshapedInput = inputTensor.reshape([1, ...inputTensor.shape]);

      result = this.model.execute(
        { image: reshapedInput },
        [
          'Openpose/MConv_Stage6_L1_5_pointwise/BatchNorm/FusedBatchNorm',
          'Openpose/MConv_Stage6_L2_5_pointwise/BatchNorm/FusedBatchNorm'
        ]) as NamedTensorMap;
    });

    this.logdata += 'Executed on ' + d0.wallMs.toString() + 'ms\n';

    for (const key in result) {
      if (result.hasOwnProperty(key)) {
        console.log(key, result[key].shape);
      }
    }
  }

}
