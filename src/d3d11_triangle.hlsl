// Vertex data structure
struct VSInput
{
    float3 position : POSITION;
    float3 color : COLOR;
};

struct VSOutput
{
    float4 position : SV_Position;
    float3 color : COLOR;
    uint view_id : SV_RenderTargetArrayIndex;
};

// Constant buffer for transformation matrices
cbuffer TransformBuffer : register(b0)
{
    row_major float4x4 viewProj[2]; // View-projection matrices for both eyes
    row_major float4x4 model;       // Model transformation matrix
};

// Vertex Shader
VSOutput VSMain(VSInput input, uint instanceId : SV_InstanceID)
{
    VSOutput output;

    // Transform position: model -> view -> projection
    float4 worldPos = mul(float4(input.position, 1.0), model);
    output.position = mul(worldPos, viewProj[instanceId]);
    output.color = input.color;
    output.view_id = instanceId;

    return output;
}

// Pixel Shader
float4 PSMain(VSOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}
