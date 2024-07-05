// en.js
export default {
    home: {
        explore: 'Explore',
        language: 'Choose Language',
        english: 'English',
        chinese: 'Chinese',
        today: 'Today',
        previous:'Previous',
        withoutAssistantTip:'The Madsys of this record has been deleted. The user can only view historical conversation information and cannot continue the conversation!',
        deleteThreadTip:'Deleting records will clear historical information~'
    },
    chat:{
        inputTip:"Send a message and chat with the Madsys ~",
    },
    explore:{
        description: "Based on Lexllama, let’s create your own Madsys~",
        configuring: "Configuring",
        completed: "Completed",
        assistantName: "Name",
        assistantDescription: "Description",
        assistantStatus: "Status",
        createAssistant: "Create New Madsys",
        deleteAssistant: "Are you sure to delete this? After deleting the Madsys, its KVCache will also be cleared simultaneously~",
    },
    config:{
        title:'Configure your Madsys',
        fileTip:"Only support text, docx, .ppt, .pdf format.",
        reConfigTip:'Reconfig Madsys needs to delete kvcache, please choose carefully',
        secletFile:'Select Files',
        outOfSize:'File size exceeds 10MB, please reselect',
        fileExist:'The file already exists, please reselect',
        createAssistant:'Assistant created successfully, click the build button to start building KVCache',
    },
    build:{
        title:'Building Logs',
        step1:'Parse uploded files',
        parsingFileStep1:'File upload and reception completed',
        parsingFileStep2:{
            parse:"Parsing",
            file:"file(s)",
            total:'total',
        },
        parsingFileStep3:'Prompt loaded, ready to generate KVCache',
        step2:'Generate KVCache',
        generateStep1:'Generate KVCache calculation plan',
        generateStep2:{
            calculate:"calculating",
            token:"tokens",
            total:'total',
        },
        generateStep3:'KVCache has been generated successfully',
        durationTime:'Duration:',
        remainTime:'Time left:',
        buildProgress:'Building Progress',
        storageUsage:'KVCache Storage Usage',
    }
}
