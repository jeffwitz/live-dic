/** Presentation-only WGSL. It never writes or interpolates the numerical field. */

/** Copy the associated current image without scalar-field sampling. */
export const LIVE_BACKGROUND_SHADER = `
      struct Config { width:u32,height:u32,gx:u32,gy:u32,stride:u32,parameters:u32,step:u32,gray:u32,
        subset:f32,spacing:f32,maximum:f32,gain:f32,view_width:f32,view_height:f32,opacity:f32,padding:f32 }
      @group(0) @binding(1) var image:texture_2d<f32>;
      @group(0) @binding(2) var<uniform> config:Config;
      @vertex fn vertex(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32> {
        let x=f32((i<<1u)&2u);let y=f32(i&2u);return vec4<f32>(x*2.0-1.0,y*2.0-1.0,0,1);
      }
      @fragment fn background(@builtin(position) p:vec4<f32>)->@location(0) vec4<f32> {
        let xy=vec2<i32>(p.xy*vec2<f32>(f32(config.width)/config.view_width,f32(config.height)/config.view_height));
        let value=textureLoad(image,clamp(xy,vec2<i32>(0),vec2<i32>(i32(config.width)-1,i32(config.height)-1)),0);
        return vec4<f32>(select(value.rgb,vec3<f32>(value.r),config.gray!=0u),1);
      }
    `;

/** Sparse glyphs, anchored at current subset centres; magnification affects tails. */
export const LIVE_GLYPH_SHADER = `
      override OUTLINE_WIDTH:f32=0.0;
      struct Field { values:array<f32> }
      struct Config { width:u32,height:u32,gx:u32,gy:u32,stride:u32,parameters:u32,step:u32,gray:u32,
        subset:f32,spacing:f32,maximum:f32,gain:f32,view_width:f32,view_height:f32,opacity:f32,colour_map:f32 }
      @group(0) @binding(0) var<storage,read> field:Field;
      @group(0) @binding(2) var<uniform> config:Config;
      @group(0) @binding(3) var palette:texture_2d<f32>;
      struct Vertex { @builtin(position) position:vec4<f32>, @location(0) colour:vec4<f32> }
      fn colour(norm:f32)->vec3<f32> {
        if config.colour_map > 0.5 {
          let index=i32(floor(clamp(norm/max(config.maximum,1e-8),0.0,1.0)*255.0+0.5));
          return textureLoad(palette,vec2<i32>(index,0),0).rgb;
        }
        let t=clamp(norm/max(config.maximum,1e-8),0.0,1.0)*2.0;
        let low=vec3<f32>(21,57,99)/255.0;let middle=vec3<f32>(31,132,138)/255.0;let high=vec3<f32>(241,206,95)/255.0;
        return select(mix(low,middle,t),mix(middle,high,t-1.0),t>=1.0);
      }
      @vertex fn glyph(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->Vertex {
        let columns=(config.gx+config.step-1u)/config.step;
        let ix=min(config.gx-1u,(instance%columns)*config.step+config.step/2u);
        let iy=min(config.gy-1u,(instance/columns)*config.step+config.step/2u);
        let base=(iy*config.gx+ix)*config.stride;
        let status=field.values[base+config.parameters];
        var output:Vertex;output.position=vec4<f32>(-2,-2,0,1);output.colour=vec4<f32>(0);
        if status == -3.0 {return output;}
        let factor=vec2<f32>(config.view_width/f32(config.width),config.view_height/f32(config.height));
        let centre=(vec2<f32>(f32(ix),f32(iy))*config.spacing+vec2<f32>((config.subset-1.0)/2.0));
        let corners=array<vec2<f32>,6>(vec2<f32>(0,-1),vec2<f32>(1,-1),vec2<f32>(1,1),vec2<f32>(0,-1),vec2<f32>(1,1),vec2<f32>(0,1));
        var point=centre*factor;
        if status < 0.5 {
          // An unavailable requested subset has no reliable deformed centre.
          // Mark its reference centre, never manufacture a displacement.
          let axis=select(vec2<f32>(.7071,.7071),vec2<f32>(.7071,-.7071),vertex>=6u);
          let normal=vec2<f32>(-axis.y,axis.x);let corner=corners[vertex%6u];
          point+=(corner.x*2.0-1.0)*axis*3.0+normal*corner.y*(.65+OUTLINE_WIDTH);
          output.colour=vec4<f32>(.63,.67,.70,config.opacity);
        } else {
          let displacement=vec2<f32>(field.values[base],field.values[base+3u]);
          let end=(centre+displacement)*factor;
          let delta=displacement*config.gain*factor;
          let distance=length(delta);output.colour=vec4<f32>(colour(length(displacement)),config.opacity);
          if distance < 1.0 {
            // A short measured vector is a dot; its colour still reports norm.
            point=end;
            if vertex<6u {let c=corners[vertex];point+=vec2<f32>(c.x*2.0-1.0,c.y)*(1.25+OUTLINE_WIDTH);}
          } else {
            let direction=delta/distance;let normal=vec2<f32>(-direction.y,direction.x);
            let head=min(distance*.4,12.0);let start=end-delta;let neck=end-direction*head;
            if vertex<6u {let c=corners[vertex];point=mix(start,neck,c.x)+normal*c.y*(2.0+OUTLINE_WIDTH);}
            else if vertex<9u {
              let points=array<vec2<f32>,3>(end,neck+normal*(head*.5+OUTLINE_WIDTH),neck-normal*(head*.5+OUTLINE_WIDTH));point=points[vertex-6u];
            } else {point=start;}
          }
        }
        if OUTLINE_WIDTH>0.0 {
          output.colour=vec4<f32>(select(vec3<f32>(.025),vec3<f32>(.98),OUTLINE_WIDTH>1.5),output.colour.a);
        }
        output.position=vec4<f32>(point.x/config.view_width*2.0-1.0,1.0-point.y/config.view_height*2.0,0,1);
        return output;
      }
      @fragment fn ink(input:Vertex)->@location(0) vec4<f32> {return input.colour;}
    `;
