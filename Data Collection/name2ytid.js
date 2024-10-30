const youtube = require('youtube-search-api');
const cliProgress = require('cli-progress');
const fs = require('fs');

let names = require('./data/artist_names.json');

async function names2ids(names) {
    let ids = [];

    const bar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
    bar.start(names.length, 0);

    for (let i = 0; i < names.length; i++) {
        const result = await youtube.GetListByKeyword(names[i], true, 2, [{ type: "channel" }]);
        if (result['items'].length === 0){
            ids = [...ids, {"yt name": names[i], "yt Title": "NaN", "yt Channel ID": "NaN"}];
            // console.log(names[i] + " has no channel.");
        }
        else {
            ids = [...ids, {"yt name": names[i], "yt Title": result['items'][0]['title'], "yt Channel ID": result['items'][0]['id']}];
        }
        bar.increment();
    }
    bar.stop();
    return ids;
}

const resultPromise = names2ids(names);

resultPromise.then(function(ids) {
    const jsonContent = JSON.stringify(ids);
    fs.writeFile("data/channel_info.json", jsonContent, 'utf8', function (err) {
        if (err) {
            return console.log(err);
        }
        console.log("The file was saved!");
    });
});