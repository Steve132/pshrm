
__constant sampler_t samplerIn = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__kernel void perpixel(
        __read_only image2d_t image,
        __write_only image2d_t image_out
    ) 
{

    const int2 pos = {get_global_id(0), get_global_id(1)};
	float4 val=read_imagef(image,samplerIn,pos);
	float tmp=val.x;
	val.x=val.z;
	val.z=tmp;
	write_imagef(image_out, pos, val);
}

