import React, { useState, useEffect, useMemo } from 'react';

// Renders the subtitles using pure CSS overlaid on the video player
export default function SubtitleRenderer({
    currentTime,
    srtEntries, // Array of { id, startTime, endTime, text }
    fontSize = 40,
    fontFamily = 'Anton',
    fontColor = '#FFFFFF',
    strokeColor = '#000000',
    strokeWidth = 3,
    bold = true,
    boxColor = '#000000',
    boxOpacity = 0,
    karaokeMode = false,
    position = 'middle', // 'top', 'middle', 'bottom'
    offsetX = 0,
    offsetY = 0,
    speakerColorMode = false,
    animation = 'none',
    isDragging = false,
    onMouseDown = undefined,
    captionDragEnabled = false
}) {
    const [activeEntry, setActiveEntry] = useState(null);

    // Parse speaker colors from SRT if speakerColorMode is on
    const SPEAKER_COLORS = ['#39FF14', '#00E5FF', '#FFC400', '#FF4D4D', '#B266FF'];

    useEffect(() => {
        if (!srtEntries || srtEntries.length === 0) {
            setActiveEntry(null);
            return;
        }

        // Find the entry that matches the current time
        // Current time is in seconds
        const match = srtEntries.find(entry => {
            // In Javascript SRT parsers, times are often in milliseconds or seconds.
            // Assuming srtEntries has startTime and endTime in SECONDS
            const start = Number(entry.startTime || entry.start || 0);
            const end = Number(entry.endTime || entry.end || 0);
            return currentTime >= start && currentTime <= end;
        });

        setActiveEntry(match || null);
    }, [currentTime, srtEntries]);

    if (!activeEntry || !activeEntry.text) return null;

    // Calculate positions
    const baseTopPercent = position === 'top' ? 20 : position === 'bottom' ? 80 : 50;

    // Handle Box Opacity
    const toRgba = (hex, alpha) => {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha / 100})`;
    };

    // Extract Karaoke Words and Speaker Colors
    let displayText = activeEntry.text;
    let activeWordIndex = -1;
    let wordsToRender = [];
    let currentLineSpeakerColor = fontColor;

    if (karaokeMode) {
        // Advanced WhisperX Word-Level Object Rendering
        if (activeEntry.words && Array.isArray(activeEntry.words)) {
            wordsToRender = activeEntry.words.map((w, idx) => {
                const isActive = currentTime >= w.start && currentTime <= w.end;
                if (isActive) activeWordIndex = idx;

                // Diarization coloring (SPEAKER_00 -> 0, SPEAKER_01 -> 1)
                let wordColor = fontColor;
                if (speakerColorMode && w.speaker) {
                    const speakerNum = parseInt(w.speaker.replace('SPEAKER_', ''), 10);
                    if (!isNaN(speakerNum)) {
                        wordColor = SPEAKER_COLORS[speakerNum % SPEAKER_COLORS.length] || fontColor;
                    }
                }

                return { text: w.word, active: isActive, color: wordColor };
            });

            // Set base color for non-active words (optional but helps cohesiveness)
            if (activeWordIndex !== -1 && wordsToRender[activeWordIndex].color !== fontColor) {
                currentLineSpeakerColor = wordsToRender[activeWordIndex].color;
            }

        } else {
            // Fallback: simple tokenization from string tags (Faster-Whisper style)
            const regex = /(<u>.*?<\/u>|\S+)/g;
            const tokens = displayText.match(regex) || [displayText];
            wordsToRender = tokens.map((t, i) => {
                const isActive = t.startsWith('<u>') && t.endsWith('</u>');
                if (isActive) activeWordIndex = i;
                return {
                    text: isActive ? t.substring(3, t.length - 4) : t,
                    active: isActive,
                    color: fontColor
                };
            });
        }
    } else {
        // Clean display (No Karaoke)
        if (speakerColorMode && activeEntry.speaker) {
            const speakerNum = parseInt(activeEntry.speaker.replace('SPEAKER_', ''), 10);
            if (!isNaN(speakerNum)) {
                currentLineSpeakerColor = SPEAKER_COLORS[speakerNum % SPEAKER_COLORS.length] || fontColor;
            }
        }
        displayText = displayText.replace(/<\/?[^>]+(>|$)/g, "");
    }

    const containerStyle = {
        position: 'absolute',
        left: `calc(50% + ${offsetX}%)`,
        top: `calc(${baseTopPercent}% + ${offsetY}%)`,
        transform: 'translate(-50%, -50%)',
        width: 'min(94%, 960px)',
        textAlign: 'center',
        pointerEvents: captionDragEnabled ? 'auto' : 'none',
        zIndex: 20
    };

    const textStyle = {
        display: 'inline-block',
        fontSize: `${Math.max(12, Math.round(fontSize * 0.58))}px`,
        fontFamily: fontFamily,
        fontWeight: bold ? 700 : 400,
        color: currentLineSpeakerColor,
        textShadow: `0 0 ${strokeWidth}px ${strokeColor}`,
        backgroundColor: boxOpacity > 0 ? toRgba(boxColor, boxOpacity) : 'transparent',
        padding: '4px 8px',
        borderRadius: '8px',
        userSelect: 'none',
        cursor: captionDragEnabled ? (isDragging ? 'grabbing' : 'grab') : 'default'
    };

    const getAnimationClass = () => {
        if (animation === 'pop') return 'animate-[subtitlePop_0.2s_ease-out]';
        if (animation === 'bounce') return 'animate-[subtitleBounce_0.3s_ease-out]';
        if (animation === 'slide') return 'animate-[subtitleSlideUp_0.2s_ease-out]';
        return '';
    };

    return (
        <div style={containerStyle}>
            <span
                style={textStyle}
                onMouseDown={onMouseDown}
                className={`${getAnimationClass()} ${isDragging ? 'opacity-80 scale-105' : ''} transition-transform`}
                key={activeEntry.id || activeEntry.text} // Key on text to trigger CSS animation on change
            >
                {karaokeMode && wordsToRender.length > 0 ? (
                    wordsToRender.map((w, idx) => {
                        return (
                            <span
                                key={idx}
                                style={{
                                    display: 'inline-block',
                                    marginRight: idx < wordsToRender.length - 1 ? '0.42em' : 0,
                                    transform: w.active ? 'scale(1.16)' : 'scale(1)',
                                    fontWeight: w.active ? 800 : (bold ? 700 : 400),
                                    color: w.active ? '#39FF14' : w.color, // Active green highlight over speaker base color
                                    textShadow: w.active ? `0 0 ${Math.max(2, strokeWidth + 1)}px ${strokeColor}` : textStyle.textShadow,
                                    transition: 'transform 100ms ease, color 100ms ease'
                                }}
                            >
                                {w.text}
                            </span>
                        );
                    })
                ) : (
                    displayText
                )}
            </span>
        </div>
    );
}
