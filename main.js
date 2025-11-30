import "https://cdn.jsdelivr.net/npm/earcut@2.2.4/dist/earcut.min.js";
const earcut = window.earcut;

// ==========================================
// 1. UI ELEMENTS
// ==========================================
const canvas = document.getElementById('canvas');
const tooltip = document.getElementById('tooltip');
const ditherBtn = document.getElementById('ditherBtn');
const boundaryBtn = document.getElementById('boundaryBtn');
const colorBtn = document.getElementById('colorBtn');
const vizSelect = document.getElementById('vizSelect');
const legendMax = document.getElementById('legendMax');
const legend75 = document.getElementById('legend75');
const legend50 = document.getElementById('legend50');
const legend25 = document.getElementById('legend25');
const legendMin = document.getElementById('legendMin');
const backBtn = document.getElementById('backBtn'); 
const legendCanvas = document.getElementById('legendCanvas');
const legendCtx = legendCanvas.getContext('2d');
const legendContainer = document.getElementById('legend');
const legendLabels = document.getElementById('legendLabels');

// Modal Elements
const infoBtn = document.getElementById('infoBtn');
const infoModal = document.getElementById('infoModal');
const closeModal = document.querySelector('.closeModal');

// ==========================================
// 2. CONSTANTS & STATE
// ==========================================
const SQ_METERS_TO_SQ_MILE = 2589988.11;
const MERCATOR_Y_SCALE = 1.15; 
const US_GEO_BOUNDS = { minLon: -125.0, maxLon: -66.0, minLat: 24.0, maxLat: 50.0 };

let ditherMode = 0; // 0: Off, 1: Bayer, 2: Blue
let showBoundaries = false;
let invertColor = 0; // 0: Standard, 1: Inverted
let vizMode = 'pop'; 
let currentView = 'national'; 
let selectedPeaId = null; 
let isPicking = false; 
let hoveredId = null; // FIX: Added missing state variable

// Camera
let camera = { x: 0, y: 0, k: 1 }; 
let isDragging = false;
let lastMouse = { x: 0, y: 0 };
let renderRequested = false;

// Data State
let currentBounds = { minX: 0, maxX: 0, minY: 0, maxY: 0 };
let nationalTotalPop = 0;
let currentViewTotalPop = 0;
let nationalBufferCache = null;

// WebGPU Globals
let device, context, format;
let peaData = [], tractData = [];
let geoIdToPea = new Map();
let blueNoiseTexture, blueNoiseSampler;
let blueNoisePixels = null; 
let buffers = {};
let renderPipeline, uniformBuffer, bindGroup;
let boundaryPipeline;
let highlightPipeline; // FIX: Added missing global
let pickingPipeline, pickingTexture, pickingBuffer, pickingBindGroup;
const idToFeature = new Map(); 
const idToLineRange = new Map(); // FIX: Added missing global

// ==========================================
// 3. ENTRY POINT
// ==========================================
run();

async function run() {
    await initWebGPU();
    await Promise.all([ loadData(), loadBlueNoise() ]);
    computePEAPopulations(); 
    
    // Initial Render
    const bounds = getNationalBounds();
    currentBounds = bounds;
    updateBuffers(peaData, bounds);
    
    // Cache National Buffers
    nationalBufferCache = { ...buffers, idToLineRange: new Map(idToLineRange) };
    
    createPickingResources(); 
    createRenderPipeline();   
    createBoundaryPipeline();
    createHighlightPipeline(); // Initialize Red Outline
    
    drawLegend();
    render();
}

// ==========================================
// 4. DATA PROCESSING
// ==========================================
async function loadData() {
    const [peaGeo, tractTopology, tractMapJson] = await Promise.all([
        fetch('data/peaFile.geojson').then(r => r.json()),
        fetch('data/tracts.json').then(r => r.json()),
        fetch('data/tract_to_pea.json').then(r => r.json())
    ]);
    
    const layerName = Object.keys(tractTopology.objects)[0];
    const tractGeo = topojson.feature(tractTopology, tractTopology.objects[layerName]);
    
    geoIdToPea = new Map();
    tractMapJson.forEach(item => {
        geoIdToPea.set(item.GEOID, item.PEA_Num);
    });

    peaData = peaGeo.features.filter(f => f.properties.PEA_Num !== 416); 
    tractData = tractGeo.features;
}

async function loadBlueNoise() {
    const response = await fetch('./assets/HDR_LA_0(256x256).png');
    const blob = await response.blob();
    const imgBitmap = await createImageBitmap(blob);
    
    blueNoiseTexture = device.createTexture({ size: [imgBitmap.width, imgBitmap.height, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    device.queue.copyExternalImageToTexture({ source: imgBitmap }, { texture: blueNoiseTexture }, [imgBitmap.width, imgBitmap.height]);
    blueNoiseSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat' });

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = imgBitmap.width;
    tempCanvas.height = imgBitmap.height;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(imgBitmap, 0, 0);
    blueNoisePixels = ctx.getImageData(0, 0, imgBitmap.width, imgBitmap.height).data;
}

function computePEAPopulations() {
    console.log("Starting Population Aggregation...");
    let matchCount = 0;
    tractData.forEach(feature => {
        const props = feature.properties;
        const id = String(props.GEOID); 
        const rawPop = props.pop || props.population || props.POPULATION || 0;
        const rawArea = props.aland || props.ALAND || props.area || 0;
        const pop = parseInt(rawPop) || 0;
        const aland = parseInt(rawArea) || 0;
        const peaNum = geoIdToPea.get(id);
        if (peaNum !== undefined) matchCount++;
        props.value = pop;       
        props.aland = aland;     
        props.peaNum = peaNum;   
    });
    console.log(`Matched ${matchCount} of ${tractData.length} tracts to PEAs.`);

    const peaPopSums = new Map(), peaAreaSums = new Map();
    tractData.forEach(f => {
        const pNum = f.properties.peaNum;
        if (pNum !== undefined) { 
            peaPopSums.set(pNum, (peaPopSums.get(pNum) || 0) + f.properties.value);
            peaAreaSums.set(pNum, (peaAreaSums.get(pNum) || 0) + f.properties.aland);
        }
    });

    nationalTotalPop = 0;
    const maxPop = Math.max(...Array.from(peaPopSums.values())) || 1;

    peaData.forEach(f => {
        const num = f.properties.PEA_Num;
        const totalPop = peaPopSums.get(num) || 0;
        f.properties.POP2020 = totalPop;
        f.properties.ALAND = peaAreaSums.get(num) || 0;
        f.properties.popNormal = (totalPop / maxPop);
        nationalTotalPop += totalPop;
    });
    console.log(`National Total Pop: ${nationalTotalPop.toLocaleString()}`);
}

// ==========================================
// 5. WEBGPU INIT & UPDATE
// ==========================================
async function initWebGPU() {
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    const adapter = await navigator.gpu.requestAdapter();
    device = await adapter.requestDevice();
    context = canvas.getContext('webgpu');
    format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });
    
    uniformBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    resizeCanvas();
    window.addEventListener('resize', () => {
        resizeCanvas();
        createPickingResources();
        refreshMapData();
    });
}

function resizeCanvas() {
    canvas.width = canvas.clientWidth * window.devicePixelRatio;
    canvas.height = canvas.clientHeight * window.devicePixelRatio;
}

function updateBuffers(features, bounds) {
    const vertices = [], colors = [], pickingColors = [], indices = [], lineIndices = [];
    let idxOffset = 0, currentId = 1; 
    idToFeature.clear();
    idToLineRange.clear();

    const getValue = (f) => {
        const pop = f.properties.POP2020 || f.properties.value || 0;
        if (vizMode === 'pop') return pop;
        const area = f.properties.ALAND || f.properties.aland || 0;
        if (area === 0) return 0;
        return pop / (area / SQ_METERS_TO_SQ_MILE);
    };

    const vals = features.map(f => getValue(f));
    const maxVal = Math.max(...vals) || 1;
    let minVal = Math.min(...vals);
    if (!isFinite(minVal)) minVal = 0;

    const suffix = (vizMode === 'pop') ? "" : "/sq mi";
    const fmt = (n) => Math.round(n).toLocaleString() + suffix;
    
    legendMax.innerText = fmt(maxVal);
    legend75.innerText = fmt(maxVal * 0.75);
    legend50.innerText = fmt(maxVal * 0.50);
    legend25.innerText = fmt(maxVal * 0.25);
    legendMin.innerText = fmt(minVal);

    features.forEach(f => {
        if (!f.geometry) return;
        idToFeature.set(currentId, f.properties);
        const idR = ((currentId >> 0) & 0xFF) / 255;
        const idG = ((currentId >> 8) & 0xFF) / 255;
        const idB = ((currentId >> 16) & 0xFF) / 255;

        const lineStart = lineIndices.length; // Start of this feature's outline

        const rings = (f.geometry.type === 'MultiPolygon') ? f.geometry.coordinates : [f.geometry.coordinates];
        rings.forEach(polygon => {
             const ring = (f.geometry.type === 'MultiPolygon') ? polygon[0] : polygon[0];
             const flat = [];
             ring.forEach((c, i) => {
                 const [x, y] = toMercator(c[0], c[1]);
                 const clipX = ((x - bounds.minX) / (bounds.maxX - bounds.minX)) * 2 - 1;
                 const clipY = ((y - bounds.minY) / (bounds.maxY - bounds.minY)) * 2 - 1;
                 flat.push(clipX, clipY);
                 lineIndices.push(idxOffset + i, idxOffset + ((i + 1) % ring.length));
             });

             const tri = earcut(flat);
             tri.forEach(ind => indices.push(ind + idxOffset));

             const val = getValue(f);
             const normalized = val / maxVal;
             const visualPop = Math.pow(normalized, 0.5); 
             const shade = 1.0 - visualPop; 

             ring.forEach(c => {
                 const [x, y] = toMercator(c[0], c[1]);
                 const clipX = ((x - bounds.minX) / (bounds.maxX - bounds.minX)) * 2 - 1;
                 const clipY = ((y - bounds.minY) / (bounds.maxY - bounds.minY)) * 2 - 1;
                 vertices.push(clipX, clipY);
                 colors.push(shade, shade, shade, 1.0);
                 pickingColors.push(idR, idG, idB, 1.0);
             });
             idxOffset += ring.length;
        });
        
        const lineCount = lineIndices.length - lineStart;
        idToLineRange.set(currentId, { start: lineStart, count: lineCount });
        
        currentId++;
    });

    buffers.vertex = device.createBuffer({ size: vertices.length * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffers.vertex, 0, new Float32Array(vertices));
    buffers.color = device.createBuffer({ size: colors.length * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffers.color, 0, new Float32Array(colors));
    buffers.pickingColor = device.createBuffer({ size: pickingColors.length * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffers.pickingColor, 0, new Float32Array(pickingColors));
    buffers.index = device.createBuffer({ size: indices.length * 4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffers.index, 0, new Uint32Array(indices));
    buffers.indexCount = indices.length;
    buffers.boundaryIndex = device.createBuffer({ size: lineIndices.length * 4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffers.boundaryIndex, 0, new Uint32Array(lineIndices));
    buffers.boundaryIndexCount = lineIndices.length;
    
    updateUniforms();
}

// ======== 6. LEGEND ========
function drawLegend() {
    const dpr = window.devicePixelRatio || 1;
    const cssH = 140; 
    legendCanvas.style.width = '20px';
    legendCanvas.style.height = cssH + 'px';
    legendLabels.style.height = cssH + 'px';

    legendCanvas.width = 20 * dpr;
    legendCanvas.height = cssH * dpr;

    const w = legendCanvas.width;
    const h = legendCanvas.height;
    const imgData = legendCtx.createImageData(w, h);
    const data = imgData.data;

    const bayer = [0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26, 12, 44, 4, 36, 14, 46, 6, 38, 60, 28, 52, 20, 62, 30, 54, 22, 3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25, 15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21];

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const intensity = y / (h - 1); 
            let normalized = 1.0 - intensity;
            if (intensity > 0.98) normalized = 0.0;

            let visualPop = Math.pow(normalized, 0.5);
            let finalInk = visualPop; 

            let threshold = 0.5;
            if (ditherMode === 1) threshold = bayer[(y % 8) * 8 + (x % 8)] / 64.0;
            else if (ditherMode === 2 && blueNoisePixels) {
                const bx = x % 256; const by = y % 256;
                const bIdx = (by * 256 + bx) * 4; 
                threshold = blueNoisePixels[bIdx] / 255.0;
            } else if (ditherMode === 0) {
                let val = 255 * (1 - finalInk); 
                if (invertColor === 1) val = 255 * finalInk; 
                const idx = (y * w + x) * 4;
                data[idx] = val; data[idx+1] = val; data[idx+2] = val; data[idx+3] = 255;
                continue;
            }

            const isInk = (1.0 - finalInk) < threshold;
            const idx = (y * w + x) * 4;
            if (isInk) {
                if (invertColor === 1) { 
                    data[idx] = 255; data[idx+1] = 255; data[idx+2] = 255; data[idx+3] = 255;
                } else { 
                    data[idx] = 0; data[idx+1] = 0; data[idx+2] = 0; data[idx+3] = 255;
                }
            } else {
                data[idx] = 0; data[idx+1] = 0; data[idx+2] = 0; data[idx+3] = 0;
            }
        }
    }
    legendCtx.putImageData(imgData, 0, 0);
}

// ==========================================
// 7. PIPELINES
// ==========================================
function createRenderPipeline() {
    const vs = `
        struct VSOut { @builtin(position) pos: vec4f, @location(0) col: vec4f, @location(1) uv: vec2f };
        struct U { params: vec4f, invert: f32 }; 
        @group(0) @binding(0) var<uniform> u: U;
        @vertex fn main(@location(0) p: vec2f, @location(1) c: vec4f) -> VSOut {
            let scale = u.params.y;
            let tx = u.params.z;
            let ty = u.params.w;
            let zoomed = (p * scale) + vec2f(tx, ty);
            return VSOut(vec4f(zoomed, 0., 1.), c, (zoomed+vec2f(1.))*.5);
        }`;
    const fs = `
        struct U { params: vec4f, invert: f32 }; 
        @group(0) @binding(0) var<uniform> u: U;
        @group(0) @binding(1) var blueTex: texture_2d<f32>;
        @group(0) @binding(2) var blueSampler: sampler;
        fn bayer(x:u32,y:u32)->f32 { var d=array<f32,64>(0.,32.,8.,40.,2.,34.,10.,42.,48.,16.,56.,24.,50.,18.,58.,26.,12.,44.,4.,36.,14.,46.,6.,38.,60.,28.,52.,20.,62.,30.,54.,22.,3.,35.,11.,43.,1.,33.,9.,41.,51.,19.,59.,27.,49.,17.,57.,25.,15.,47.,7.,39.,13.,45.,5.,37.,63.,31.,55.,23.,61.,29.,53.,21.); return d[(y%8)*8+(x%8)]/64.; }
        @fragment fn main(@builtin(position) fc: vec4f, @location(0) c: vec4f) -> @location(0) vec4f {
            var th=0.5;
            let dither = u.params.x;
            let val = c.r; 
            var isInk = false;
            
            if(dither > 0.9 && dither < 1.1) { th=bayer(u32(fc.x), u32(fc.y)); if(val<th){ isInk=true; } }
            else if(dither > 1.9) { let uv=vec2f(fc.x/256.0, fc.y/256.0); th=textureSample(blueTex, blueSampler, uv).r; if(val<th){ isInk=true; } }
            else {
                if (u.invert > 0.5) { return vec4f(vec3f(1.0 - val), 1.0); }
                else { return vec4f(vec3f(val), 1.0); }
            }

            if (u.invert > 0.5) { 
                if (isInk) { return vec4f(1.,1.,1.,1.); } else { return vec4f(0.,0.,0.,0.); } 
            } else { 
                if (isInk) { return vec4f(0.,0.,0.,1.); } else { return vec4f(1.,1.,1.,0.); } 
            }
        }`;
    
    const bgLayout = device.createBindGroupLayout({ entries: [{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:'uniform',minBindingSize:32}}, {binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'float'}}, {binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:'filtering'}}] });
    renderPipeline = device.createRenderPipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [bgLayout] }), vertex: { module: device.createShaderModule({code:vs}), entryPoint:'main', buffers:[{arrayStride:8,attributes:[{shaderLocation:0,offset:0,format:'float32x2'}]},{arrayStride:16,attributes:[{shaderLocation:1,offset:0,format:'float32x4'}]}] }, fragment: { module: device.createShaderModule({code:fs}), entryPoint:'main', targets:[{format}] }, primitive: { topology: 'triangle-list' } });
    bindGroup = device.createBindGroup({ layout: bgLayout, entries: [{binding:0,resource:{buffer:uniformBuffer}}, {binding:1,resource:blueNoiseTexture.createView()}, {binding:2,resource:blueNoiseSampler}] });
    updateUniforms();
}

function createBoundaryPipeline() {
    const vs = `
        struct U { params: vec4f, invert: f32 };
        @group(0) @binding(0) var<uniform> u: U;
        @vertex fn main(@location(0) p: vec2f)->@builtin(position) vec4f { 
            let scale = u.params.y;
            let tx = u.params.z;
            let ty = u.params.w;
            let zoomed = (p * scale) + vec2f(tx, ty);
            return vec4f(zoomed, 0., 1.); 
        }`;
    const fs = `
        struct U { params: vec4f, invert: f32 };
        @group(0) @binding(0) var<uniform> u: U;
        @fragment fn main()->@location(0) vec4f { 
            if (u.invert > 0.5) { return vec4f(0.4, 0.6, 0.8, 0.5); }
            return vec4f(0.4, 0.6, 0.8, 1.0); 
        }`;
    const bgLayout = device.createBindGroupLayout({ entries: [{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:'uniform',minBindingSize:32}}, {binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'float'}}, {binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:'filtering'}}] });
    boundaryPipeline = device.createRenderPipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [bgLayout] }), vertex: { module: device.createShaderModule({code:vs}), entryPoint:'main', buffers:[{arrayStride:8,attributes:[{shaderLocation:0,offset:0,format:'float32x2'}]}] }, fragment: { module: device.createShaderModule({code:fs}), entryPoint:'main', targets:[{format}] }, primitive: { topology: 'line-list' } });
}

// NEW: Create Highlight Pipeline (Red Outline)
function createHighlightPipeline() {
    const vs = `
        struct U { params: vec4f, invert: f32 };
        @group(0) @binding(0) var<uniform> u: U;
        @vertex fn main(@location(0) p: vec2f)->@builtin(position) vec4f { 
            let scale = u.params.y;
            let tx = u.params.z;
            let ty = u.params.w;
            let zoomed = (p * scale) + vec2f(tx, ty);
            return vec4f(zoomed, 0., 1.); 
        }`;
    const fs = `
        @fragment fn main()->@location(0) vec4f { 
            return vec4f(1.0, 0.0, 0.0, 1.0); // Solid Red
        }`;
    // Reuse boundary layout since inputs are same
    const bgLayout = device.createBindGroupLayout({ entries: [{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:'uniform',minBindingSize:32}}, {binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'float'}}, {binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:'filtering'}}] });
    highlightPipeline = device.createRenderPipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [bgLayout] }), vertex: { module: device.createShaderModule({code:vs}), entryPoint:'main', buffers:[{arrayStride:8,attributes:[{shaderLocation:0,offset:0,format:'float32x2'}]}] }, fragment: { module: device.createShaderModule({code:fs}), entryPoint:'main', targets:[{format}] }, primitive: { topology: 'line-list' } });
}

function createPickingResources() {
    pickingTexture = device.createTexture({ size: [canvas.width, canvas.height], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    pickingBuffer = device.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const vsFull = `
        struct VSOut { @builtin(position) p: vec4f, @location(0) c: vec4f }; 
        struct U { params: vec4f, invert: f32 };
        @group(0) @binding(0) var<uniform> u: U;
        @vertex fn main(@location(0) p: vec2f, @location(1) c: vec4f)->VSOut { 
            let scale = u.params.y;
            let tx = u.params.z;
            let ty = u.params.w;
            let zoomed = (p * scale) + vec2f(tx, ty);
            return VSOut(vec4f(zoomed, 0., 1.), c); 
        }`;
    const fs = `@fragment fn main(@location(0) c: vec4f)->@location(0) vec4f { return c; }`;
    const pickingBGLayout = device.createBindGroupLayout({ entries: [{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:'uniform',minBindingSize:32}}] });
    pickingPipeline = device.createRenderPipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [pickingBGLayout] }), vertex: { module: device.createShaderModule({code:vsFull}), entryPoint:'main', buffers:[{arrayStride:8,attributes:[{shaderLocation:0,offset:0,format:'float32x2'}]},{arrayStride:16,attributes:[{shaderLocation:1,offset:0,format:'float32x4'}]}] }, fragment: { module: device.createShaderModule({code:fs}), entryPoint:'main', targets:[{format:'rgba8unorm'}] }, primitive: { topology: 'triangle-list' } });
    pickingBindGroup = device.createBindGroup({ layout: pickingBGLayout, entries: [{binding:0,resource:{buffer:uniformBuffer}}] });
}

function render() {
    if (!device || !renderPipeline || !buffers.index) return;
    const encoder = device.createCommandEncoder();
    
    let clearVal = { r: 0.68, g: 0.85, b: 0.90, a: 1.0 }; 
    if (invertColor === 1) clearVal = { r: 0.1, g: 0.1, b: 0.18, a: 1.0 }; 

    const pass = encoder.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', clearValue: clearVal, storeOp: 'store' }] });
    
    // 1. Draw Map
    pass.setPipeline(renderPipeline); pass.setBindGroup(0, bindGroup); pass.setVertexBuffer(0, buffers.vertex); pass.setVertexBuffer(1, buffers.color); pass.setIndexBuffer(buffers.index, 'uint32'); pass.drawIndexed(buffers.indexCount);
    
    // 2. Draw Highlight (if hovered)
    if (hoveredId !== null && highlightPipeline) {
        const range = idToLineRange.get(hoveredId);
        if (range) {
            pass.setPipeline(highlightPipeline);
            pass.setBindGroup(0, bindGroup); // Re-use Main BindGroup for highlight
            pass.setVertexBuffer(0, buffers.vertex); 
            pass.setIndexBuffer(buffers.boundaryIndex, 'uint32'); 
            pass.drawIndexed(range.count, 1, range.start);
        }
    }

    // 3. Draw Boundaries
    if (showBoundaries && boundaryPipeline) { pass.setPipeline(boundaryPipeline); pass.setBindGroup(0, bindGroup); pass.setVertexBuffer(0, buffers.vertex); pass.setIndexBuffer(buffers.boundaryIndex, 'uint32'); pass.drawIndexed(buffers.boundaryIndexCount); }
    pass.end(); device.queue.submit([encoder.finish()]);
}

function updateUniforms() {
    if (!uniformBuffer) return;
    const data = new Float32Array([ditherMode, camera.k, camera.x, camera.y, invertColor, 0, 0, 0]);
    device.queue.writeBuffer(uniformBuffer, 0, data);
}

function requestRender() {
    if (!renderRequested) {
        renderRequested = true;
        requestAnimationFrame(() => {
            updateUniforms();
            render();
            renderRequested = false;
        });
    }
}

// ======== 8. INTERACTIONS & UTILS ========
function getNationalBounds() {
    const [minX, minY] = toMercator(US_GEO_BOUNDS.minLon, US_GEO_BOUNDS.minLat);
    const [maxX, maxY] = toMercator(US_GEO_BOUNDS.maxLon, US_GEO_BOUNDS.maxLat);
    let bounds = { minX, maxX, minY, maxY };
    bounds = adjustBoundsToScreen(bounds);
    const height = bounds.maxY - bounds.minY;
    bounds.maxY += height * 0.20; 
    return bounds;
}

function toMercator(lon, lat) {
    const MAX_LAT = 85.051129;
    lat = Math.max(Math.min(MAX_LAT, lat), -MAX_LAT);
    const x = (lon * Math.PI) / 180;
    let y = Math.log(Math.tan((Math.PI / 4) + (lat * Math.PI / 360)));
    y = y * MERCATOR_Y_SCALE;
    return [x, y];
}

function adjustBoundsToScreen(bounds) {
    const canvasAspect = canvas.width / canvas.height;
    const mapWidth = bounds.maxX - bounds.minX;
    const mapHeight = bounds.maxY - bounds.minY;
    const mapAspect = mapWidth / mapHeight;
    let newMinX = bounds.minX, newMaxX = bounds.maxX, newMinY = bounds.minY, newMaxY = bounds.maxY;

    if (canvasAspect > mapAspect) {
        const targetWidth = mapHeight * canvasAspect;
        const delta = targetWidth - mapWidth;
        newMinX -= delta / 2; newMaxX += delta / 2;
    } else {
        const targetHeight = mapWidth / canvasAspect;
        const delta = targetHeight - mapHeight;
        newMinY -= delta / 2; newMaxY += delta / 2;
    }
    return { minX: newMinX, maxX: newMaxX, minY: newMinY, maxY: newMaxY };
}

// Event Listeners
infoBtn.onclick = () => infoModal.style.display = "block";
closeModal.onclick = () => infoModal.style.display = "none";
window.onclick = (e) => { if (e.target == infoModal) infoModal.style.display = "none"; };

ditherBtn.onclick = () => {
    ditherMode = (ditherMode + 1) % 3;
    const labels = ["Dither: Off", "Dither: Bayer", "Dither: Blue Noise"];
    ditherBtn.innerText = labels[ditherMode];
    drawLegend(); 
    updateUniforms();
    render();
};

boundaryBtn.onclick = () => {
    showBoundaries = !showBoundaries;
    boundaryBtn.innerText = `Boundaries: ${showBoundaries ? 'On' : 'Off'}`;
    render();
};

colorBtn.onclick = () => {
    invertColor = invertColor === 0 ? 1 : 0;
    colorBtn.innerText = invertColor === 0 ? "Color: Light" : "Color: Dark";
    
    if (invertColor === 1) {
        legendContainer.style.backgroundColor = "rgba(0, 0, 0, 0.85)";
        legendLabels.style.color = "#eee";
    } else {
        legendContainer.style.backgroundColor = "rgba(255, 255, 255, 0.95)";
        legendLabels.style.color = "#333";
    }
    
    drawLegend(); 
    updateUniforms();
    render();
};

vizSelect.onchange = (e) => {
    vizMode = e.target.value;
    nationalBufferCache = null; 
    refreshMapData();
};

backBtn.onclick = () => {
    currentView = 'national';
    selectedPeaId = null; 
    backBtn.style.display = 'none';
    camera = { x: 0, y: 0, k: 1 };
    updateUniforms();
    refreshMapData();
};

function refreshMapData() {
    if (currentView === 'national') {
        const bounds = getNationalBounds();
        currentBounds = bounds;
        
        if (nationalBufferCache) {
            buffers = { ...nationalBufferCache };
            idToLineRange.clear();
            nationalBufferCache.idToLineRange.forEach((v,k) => idToLineRange.set(k,v));
            updateBuffers(peaData, bounds); 
        } else {
            updateBuffers(peaData, bounds);
            nationalBufferCache = { ...buffers, idToLineRange: new Map(idToLineRange) }; 
        }
    } else {
        const bounds = adjustBoundsToScreen(currentBounds);
        const data = tractData.filter(f => String(f.properties.peaNum) === String(selectedPeaId));
        updateBuffers(data, bounds);
    }
    render();
}

function enterDetailView(peaNum) {
    const targetPea = String(peaNum);
    const relevantTracts = tractData.filter(f => String(f.properties.peaNum) === targetPea);
    if (relevantTracts.length === 0) return;

    currentViewTotalPop = relevantTracts.reduce((acc, f) => acc + (f.properties.value || 0), 0);

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    relevantTracts.forEach(f => {
        if (!f.geometry) return;
        const rings = (f.geometry.type === 'MultiPolygon') ? f.geometry.coordinates : [f.geometry.coordinates];
        rings.forEach(poly => {
            const ring = (f.geometry.type === 'MultiPolygon') ? poly[0] : poly[0];
            ring.forEach(c => {
                const [mx, my] = toMercator(c[0], c[1]);
                if (mx < minX) minX = mx;
                if (mx > maxX) maxX = mx;
                if (my < minY) minY = my;
                if (my > maxY) maxY = my;
            });
        });
    });

    const padX = (maxX - minX) * 0.05;
    const padY = (maxY - minY) * 0.05;
    const rawBounds = { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
    const correctedBounds = adjustBoundsToScreen(rawBounds);
    const height = correctedBounds.maxY - correctedBounds.minY;
    correctedBounds.maxY += height * 0.15; 

    camera = { x: 0, y: 0, k: 1 };
    updateUniforms();

    currentView = 'detail';
    selectedPeaId = peaNum; 
    currentBounds = correctedBounds;
    backBtn.style.display = 'block';

    updateBuffers(relevantTracts, correctedBounds);
    render();
}

async function identifyFeature(clientX, clientY) {
    if (isPicking || !pickingPipeline || !pickingTexture) return null;
    isPicking = true;
    try {
        const rect = canvas.getBoundingClientRect();
        const x = Math.floor((clientX - rect.left) * window.devicePixelRatio);
        const y = Math.floor((clientY - rect.top) * window.devicePixelRatio);
        if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return null;

        const commandEncoder = device.createCommandEncoder();
        const pass = commandEncoder.beginRenderPass({ colorAttachments: [{ view: pickingTexture.createView(), loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }] });
        pass.setPipeline(pickingPipeline); pass.setBindGroup(0, pickingBindGroup); pass.setVertexBuffer(0, buffers.vertex); pass.setVertexBuffer(1, buffers.pickingColor); pass.setIndexBuffer(buffers.index, 'uint32'); pass.drawIndexed(buffers.indexCount);
        pass.end();

        commandEncoder.copyTextureToBuffer({ texture: pickingTexture, origin: { x, y } }, { buffer: pickingBuffer, bytesPerRow: 256 }, { width: 1, height: 1 });
        device.queue.submit([commandEncoder.finish()]);

        await pickingBuffer.mapAsync(GPUMapMode.READ);
        const data = new Uint8Array(pickingBuffer.getMappedRange());
        const id = data[0] + (data[1] << 8) + (data[2] << 16);
        pickingBuffer.unmap();
        
        if (id !== hoveredId) {
            hoveredId = id > 0 ? id : null;
            requestRender(); 
        }
        
        return idToFeature.get(id);
    } catch (e) { return null; } finally { isPicking = false; }
}

canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const zoomIntensity = 0.003; 
    const delta = -e.deltaY * zoomIntensity;
    const oldK = camera.k;
    const newK = Math.min(Math.max(oldK * (1 + delta), 0.5), 150);
    const rect = canvas.getBoundingClientRect();
    const mouseX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const mouseY = -(((e.clientY - rect.top) / rect.height) * 2 - 1); 
    camera.x += (mouseX - camera.x) * (1 - newK / oldK);
    camera.y += (mouseY - camera.y) * (1 - newK / oldK);
    camera.k = newK;
    requestRender();
}, { passive: false });

canvas.addEventListener('mousedown', e => { isDragging = true; lastMouse = { x: e.clientX, y: e.clientY }; });
window.addEventListener('mouseup', () => { isDragging = false; });

canvas.addEventListener('mousemove', async (e) => {
    const feature = await identifyFeature(e.clientX, e.clientY);
    canvas.style.cursor = feature ? 'pointer' : (isDragging ? 'grabbing' : 'default');

    if (isDragging) {
        const dx = (e.clientX - lastMouse.x) / canvas.clientWidth * 2;
        const dy = -(e.clientY - lastMouse.y) / canvas.clientHeight * 2; 
        camera.x += dx; camera.y += dy;
        lastMouse = { x: e.clientX, y: e.clientY };
        requestRender();
        return; 
    }

    if (feature) {
        tooltip.style.display = 'block';
        const label = currentView === 'national' ? `PEA #${feature.PEA_Num}` : `Census Tract`;
        let name = feature.PEA_Name || feature.NAME || 'Unknown';
        const pop = feature.POP2020 || feature.value || 0;
        const area = feature.ALAND || feature.aland || 0;
        const sqMiles = area / SQ_METERS_TO_SQ_MILE;
        const density = sqMiles > 0 ? (pop / sqMiles) : 0;

        let subText = "";
        if (vizMode === 'pop') {
            let percentage = "0%";
            if (currentView === 'national' && nationalTotalPop > 0) percentage = ((pop / nationalTotalPop) * 100).toFixed(2) + "%";
            else if (currentView === 'detail' && currentViewTotalPop > 0) percentage = ((pop / currentViewTotalPop) * 100).toFixed(2) + "%";
            subText = `<span style="color: #4db8ff;">${percentage} of Total</span>`;
        } else {
            subText = `<span style="color: #4db8ff;">${Math.round(density).toLocaleString()} / sq mi</span>`;
        }
        
        tooltip.innerHTML = `<strong>${name}</strong><br/><span style="font-size: 0.9em; color: #ccc;">${label}</span><br/>Pop: ${Math.round(pop).toLocaleString()}<br/>${subText}`;
        const tipWidth = tooltip.offsetWidth; const tipHeight = tooltip.offsetHeight; const gap = 15; 
        let left = e.clientX + gap; let top = e.clientY + gap;
        if (left + tipWidth > window.innerWidth) left = e.clientX - tipWidth - gap;
        if (top + tipHeight > window.innerHeight) top = e.clientY - tipHeight - gap;
        tooltip.style.left = left + 'px'; tooltip.style.top = top + 'px';
    } else {
        tooltip.style.display = 'none';
        if (hoveredId !== null) { hoveredId = null; requestRender(); }
    }
});

canvas.addEventListener('click', async (e) => {
    if (isDragging) return; 
    if (currentView === 'detail') return; 
    const feature = await identifyFeature(e.clientX, e.clientY);
    if (feature && feature.PEA_Num) enterDetailView(feature.PEA_Num);
});