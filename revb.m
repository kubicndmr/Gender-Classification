dataset = ls('dataset/*.wav');

for s=1:length(dataset)
    speech = strcat('dataset/',dataset(s,:));
    [x, fs] = audioread(speech);
    revb_x = Reverberation(x,fs,0.01);
    revb_speech = strcat('revb/',dataset(s,:));
    audiowrite(revb_speech,revb_x,fs);
end