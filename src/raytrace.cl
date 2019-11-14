

static inline float3 delatlong(float2 posdex,const float S)
{
	posdex+=0.5f;
	posdex*=S;
	float2 sct;
	float v;
	sct.x=sincos(posdex.y,&v);
	sct.y=v;
	float2 scp;
	scp.x=sincos(posdex.x,&v);
	scp.y=v;
	return (float3)(sct.x*scp.y,sct.x*scp.x,sct.y);
}

__constant sampler_t samplerIn = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
__constant sampler_t ksamplerIn = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
__kernel void perpixel(
        __read_only image2d_t image,
        __write_only image2d_t image_out,
		__read_only image1d_t kernel_image,
		uint2 lower_img_bounds,
		uint2 upper_img_bounds
    ) 
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
	const int2 sz =  get_image_dim(image);
	const int2 osz = get_image_dim(image_out);
	const float S_scale=M_PI_F/((float)sz.y);
	
	if(pos.x >= osz.x || pos.y >= osz.y)
	{
		return;
	}
	
	const float3 this_vec=delatlong(convert_float2(pos),S_scale);
	
	float4 avg=(float4)(0.0f,0.0f,0.0f,0.0f);

	const float sample_scalex=1.f/(float)(sz.x);
	const float sample_scaley=1.f/(float)(sz.y);
	const float sample_scale=sample_scalex*sample_scaley/(4.0*M_PI_F);

	for(int yi=lower_img_bounds.y;yi<upper_img_bounds.y;yi++)
	for(int xi=lower_img_bounds.x;xi<upper_img_bounds.x;xi++)
	{
		float4 oval=read_imagef(image,samplerIn,(int2)(xi,yi));
		float2 posf=convert_float2((int2)(xi,yi));
		float cweight=sin((posf.y+0.5f)*S_scale);
		oval*=cweight;
		const float3 sample_vec=delatlong(posf,S_scale);
		float dotv=dot(sample_vec,this_vec);
		dotv=max(dotv,0.0f);
		oval*=read_imagef(kernel_image,ksamplerIn,1.0f-dotv);
		oval*=sample_scale;
		avg+=oval;
	}
	
	write_imagef(image_out, pos,avg);
}

