struct Vertex { position: vec4<f32>, normal: vec4<f32> }
struct Camera { view: vec4<f32>, style: vec4<f32> }
@group(0) @binding(0) var<storage,read> geometry: array<Vertex>;
@group(0) @binding(1) var<uniform> camera: Camera;
struct Out { @builtin(position) position: vec4<f32>, @location(0) normal: vec3<f32>, @location(1) bary: vec3<f32> }
fn rotate(p: vec3<f32>) -> vec3<f32> {
  let cy=cos(camera.view.y); let sy=sin(camera.view.y);
  let cp=cos(camera.view.z); let sp=sin(camera.view.z);
  let q=vec3<f32>(cy*p.x+sy*p.z,p.y,-sy*p.x+cy*p.z);
  return vec3<f32>(q.x,cp*q.y-sp*q.z,sp*q.y+cp*q.z);
}
@vertex fn vs(@builtin(vertex_index) index:u32)->Out {
  let p=rotate(geometry[index].position.xyz);
  let z=camera.view.w-p.z;
  var out: Out;
  out.position=vec4<f32>(p.x*2.41421356/camera.view.x,p.y*2.41421356,z*10.0/9.99-0.01*10.0/9.99,z);
  out.normal=rotate(geometry[index].normal.xyz);
  out.bary=vec3<f32>(0.0); out.bary[index%3u]=1.0;
  return out;
}
@fragment fn fs(input:Out)->@location(0) vec4<f32> {
  let light=0.3+0.7*abs(dot(normalize(input.normal),normalize(vec3<f32>(0.4,0.7,1.0))));
  let width=fwidth(input.bary)*1.2;
  let edge=smoothstep(vec3<f32>(0.0),width,input.bary);
  let interior=min(edge.x,min(edge.y,edge.z));
  let base=mix(vec3<f32>(0.13,0.7,0.64),vec3<f32>(0.48,0.37,0.89),camera.style.y);
  if camera.style.x==2.0 && interior>0.7 { discard; }
  let color=base*light;
  if camera.style.x>0.0 { return vec4<f32>(mix(vec3<f32>(0.78,0.94,0.94),color,interior),1.0); }
  return vec4<f32>(color,1.0);
}
