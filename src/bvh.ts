/** Stackless 4D BVH forest. Nodes: low vec4, high vec4, escape/start/count/root u32. */
export function buildBvh(positions: Float32Array, cells: Uint32Array) {
  const count = cells.length / 8;
  const bounds = new Float32Array(count * 8);
  const order = Array.from({length: count}, (_,i) => i);
  for (let id=0; id<count; id++) {
    for (let axis=0; axis<4; axis++) {
      let low=Infinity, high=-Infinity;
      for (let corner=0; corner<4; corner++) {
        const value=positions[cells[id*8+corner]*4+axis];
        low=Math.min(low,value); high=Math.max(high,value);
      }
      bounds.set([low,high],id*8+axis*2);
    }
  }
  function split(start: number, end: number) {
    const low=[Infinity,Infinity,Infinity,Infinity], high=low.map(()=>-Infinity);
    for(let i=start;i<end;i++)for(let axis=0;axis<4;axis++) {
      const at=order[i]*8+axis*2;
      const center=bounds[at]*0.5+bounds[at+1]*0.5;
      low[axis]=Math.min(low[axis],center);high[axis]=Math.max(high[axis],center);
    }
    let axis=0;for(let j=1;j<4;j++)if(high[j]-low[j]>high[axis]-low[axis])axis=j;
    const sorted=order.slice(start,end).sort((a,b)=>
      (bounds[a*8+axis*2]*0.5+bounds[a*8+axis*2+1]*0.5)
      -(bounds[b*8+axis*2]*0.5+bounds[b*8+axis*2+1]*0.5) || a-b);
    for(let i=0;i<sorted.length;i++)order[start+i]=sorted[i];
    return (start+end)>>>1;
  }
  const ranges: [number,number][]=[];
  function partition(start: number,end: number) {
    if(end-start<=256){ranges.push([start,end]);return;}
    const mid=split(start,end);partition(start,mid);partition(mid,end);
  }
  partition(0,count);
  const records: {low:number[];high:number[];escape:number;start:number;count:number;root:number}[]=
    ranges.map(()=>({low:[0,0,0,0],high:[0,0,0,0],escape:0,start:0,count:0,root:0}));
  function tree(start:number,end:number):number {
    const index=records.length;
    const node={low:[Infinity,Infinity,Infinity,Infinity],high:[-Infinity,-Infinity,-Infinity,-Infinity],
      escape:0,start,count:0,root:0};records.push(node);
    for(let i=start;i<end;i++)for(let axis=0;axis<4;axis++) {
      const at=order[i]*8+axis*2;
      node.low[axis]=Math.min(node.low[axis],bounds[at]);
      node.high[axis]=Math.max(node.high[axis],bounds[at+1]);
    }
    if(end-start<=16)node.count=end-start;
    else {const mid=split(start,end);tree(start,mid);tree(mid,end);}
    node.escape=records.length;
    return index;
  }
  ranges.forEach(([start,end],i)=>{records[i].root=tree(start,end);});
  const raw=new ArrayBuffer(Math.max(48,records.length*48));
  const floats=new Float32Array(raw),words=new Uint32Array(raw);
  records.forEach((node,i)=>{
    floats.set(node.low,i*12);floats.set(node.high,i*12+4);
    words.set([node.escape,node.start,node.count,node.root],i*12+8);
  });
  return {nodes:new Uint32Array(raw),order:new Uint32Array(order),roots:ranges.length};
}
