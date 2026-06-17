import { useState } from 'react';

// Require the explicit module format since mediabunny might be pure ESM or CJS.
import {
    Input,
    Output,
    CanvasSource,
    AudioSampleSink,
    AudioSampleSource,
    Mp4OutputFormat,
    StreamTarget,
    BlobSource,
    VideoSampleSink,
    ALL_FORMATS
} from 'mediabunny';

export function useLocalBurn() {
    const [isExporting, setIsExporting] = useState(false);
    const [exportProgress, setExportProgress] = useState(0);
    const [exportStatus, setExportStatus] = useState('');

    const startExport = async (videoElement, srtEntries, styleOpts) => {
        if (!videoElement || !videoElement.src) return;
        setIsExporting(true);
        setExportProgress(0);
        setExportStatus('Iniciando...');

        try {
            const videoBlob = await fetch(videoElement.src).then(r => r.blob());
            const input = new Input({
                source: new BlobSource(videoBlob),
                formats: ALL_FORMATS
            });

            const duration = await input.computeDuration();
            const originalVideoTrack = await input.getPrimaryVideoTrack();
            const originalAudioTrack = await input.getPrimaryAudioTrack();
            
            const canvas = document.createElement("canvas");
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
            const ctx = canvas.getContext("2d", { alpha: false });

            let outputBuffer = new Uint8Array(4 * 1024 * 1024);
            let outputSize = 0;
            const streamTarget = new StreamTarget(
                new WritableStream({
                    write(chunk) {
                        const end = chunk.position + chunk.data.byteLength;
                        if (end > outputBuffer.byteLength) {
                            let newLen = outputBuffer.byteLength;
                            while (newLen < end) newLen *= 2;
                            const grown = new Uint8Array(newLen);
                            grown.set(outputBuffer);
                            outputBuffer = grown;
                        }
                        outputBuffer.set(chunk.data, chunk.position);
                        if (end > outputSize) outputSize = end;
                    }
                }),
                { chunked: true }
            );

            const output = new Output({
                format: new Mp4OutputFormat({ fastStart: "in-memory" }),
                target: streamTarget
            });

            const exportFps = 30;
            const videoSource = new CanvasSource(canvas, {
                codec: 'avc',
                bitrate: 4000000 
            });
            output.addVideoTrack(videoSource, { frameRate: exportFps });

            let audioSource = null;
            if (originalAudioTrack) {
                audioSource = new AudioSampleSource({ codec: "mp4a", bitrate: 128000 });
                try {
                    output.addAudioTrack(audioSource);
                } catch(e) {
                    console.warn("Could not add audio track", e);
                }
            }

            await output.start();

            let videoSampleSink = null;
            if (originalVideoTrack && await originalVideoTrack.canDecode()) {
                videoSampleSink = new VideoSampleSink(originalVideoTrack);
            }

            if (originalAudioTrack && audioSource) {
                const audioSampleSink = new AudioSampleSink(originalAudioTrack);
                (async () => {
                    for await (const audioSample of audioSampleSink.samples(0, duration)) {
                        await audioSource.add(audioSample);
                        audioSample.close();
                    }
                    audioSource.close();
                })();
            }

            const totalFrames = Math.ceil(duration * exportFps);
            const drawFrameFromVideoElement = async (time) => {
                videoElement.currentTime = time;
                await new Promise((resolve) => {
                    const onSeeked = () => {
                        videoElement.removeEventListener("seeked", onSeeked);
                        resolve();
                    };
                    videoElement.addEventListener("seeked", onSeeked);
                    if (Math.abs(videoElement.currentTime - time) < 0.01) {
                        videoElement.removeEventListener("seeked", onSeeked);
                        resolve();
                    }
                });
                ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            };

            const sampleIterator = videoSampleSink ? videoSampleSink.samplesAtTimestamps((async function* () {
                for (let i = 0; i < totalFrames; i++) yield i / exportFps;
            })()) : null;

            for (let frameIndex = 0; frameIndex < totalFrames; frameIndex++) {
                const time = frameIndex / exportFps;
                setExportProgress(Math.min(100, (frameIndex / totalFrames) * 100));
                setExportStatus(`Renderizando frame ${frameIndex+1}/${totalFrames}`);

                ctx.clearRect(0, 0, canvas.width, canvas.height);

                let drewSource = false;
                if (videoSampleSink && sampleIterator) {
                    try {
                        const { value: sample } = await sampleIterator.next();
                        if (sample) {
                            sample.draw(ctx, 0, 0, canvas.width, canvas.height);
                            drewSource = true;
                            sample.close();
                        }
                    } catch (e) {}
                }
                
                if (!drewSource) {
                    await drawFrameFromVideoElement(time);
                }

                // SUBTITLE RENDER SCRIPT (Simplified layout logic)
                const activeEntry = srtEntries.find(e => time >= e.start && time <= e.end);
                if (activeEntry && activeEntry.text) {
                    ctx.save();
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    const scaleFactor = canvas.width / 500;
                    const fSize = styleOpts.fontSize || 40;
                    ctx.font = `${styleOpts.bold ? 'bold' : 'normal'} ${fSize * scaleFactor}px "${styleOpts.fontFamily || 'Anton'}"`;
                    ctx.fillStyle = styleOpts.fontColor || '#FFF';
                    
                    const pbxColor = styleOpts.boxColor || '#000';
                    const opacity = styleOpts.boxOpacity || 0;
                    
                    const px = canvas.width / 2;
                    let py = canvas.height * 0.8;
                    if (styleOpts.position === 'top') py = canvas.height * 0.2;
                    if (styleOpts.position === 'middle') py = canvas.height * 0.5;

                    const textW = ctx.measureText(activeEntry.text).width;
                    const textH = (fSize * scaleFactor) * 1.2;

                    if (opacity > 0) {
                        ctx.fillStyle = `rgba(${parseInt(pbxColor.slice(1,3),16)},${parseInt(pbxColor.slice(3,5),16)},${parseInt(pbxColor.slice(5,7),16)},${opacity/100})`;
                        ctx.fillRect(px - textW/2 - 10, py - textH/2 - 5, textW + 20, textH + 10);
                        ctx.fillStyle = styleOpts.fontColor || '#FFF';
                    }

                    if (styleOpts.strokeWidth > 0) {
                        ctx.strokeStyle = styleOpts.strokeColor || '#000';
                        ctx.lineWidth = (styleOpts.strokeWidth || 0) * scaleFactor;
                        ctx.strokeText(activeEntry.text, px, py);
                    }
                    ctx.fillText(activeEntry.text, px, py);
                    ctx.restore();
                }

                await videoSource.add(time, 1 / exportFps);
            }

            await videoSource.close();
            await output.wait();
            
            const fileBlob = new Blob([outputBuffer.subarray(0, outputSize)], { type: "video/mp4" });
            const url = URL.createObjectURL(fileBlob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `openshorts-local-${Date.now()}.mp4`;
            a.click();
            URL.revokeObjectURL(url);
            
            setExportStatus('¡Proceso terminado!');
        } catch (err) {
            console.error(err);
            setExportStatus('Hubo un error al exportar: ' + err.message);
        } finally {
            setIsExporting(false);
            setTimeout(() => setExportStatus(''), 3000);
        }
    };

    return { startExport, isExporting, exportProgress, exportStatus };
}
