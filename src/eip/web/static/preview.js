// Source preview owns one bounded transport and playback clock across algorithm presets.
export class PreviewEngine {
  constructor({read,generation,state}){this.read=read;this.generation=generation;this.state=state;this.chain=Promise.resolve();this.run=0;this.playing=false;}
  request(source,value,generation=this.generation()){
    const task=this.chain.catch(()=>{}).then(()=>generation===this.generation()?this.read(source,value):null);
    this.chain=task;return task;
  }
  stop(){this.playing=false;this.run++;this.state(false);}
  async play({advance,token,interval}){
    if(this.playing){this.stop();return;}
    this.playing=true;const run=++this.run;this.state(true);
    try{
      while(this.playing&&run===this.run){
        const started=performance.now(),before=token();await advance();
        if(run!==this.run)break;
        if(token()===before){this.stop();break;}
        await new Promise(resolve=>setTimeout(resolve,Math.max(0,interval()-(performance.now()-started))));
      }
    }catch(error){if(run===this.run)this.stop();throw error;}
  }
}
