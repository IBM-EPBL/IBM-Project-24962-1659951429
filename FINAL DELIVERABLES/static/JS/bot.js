window.watsonAssistantChatOptions = {
  integrationID: "a3d08917-7818-4e6b-bfc5-c71426f0321d", // The ID of this integration.
  region: "us-south", // The region your integration is hosted in.
  serviceInstanceID: "94816793-9df2-4149-8235-e0686d6a1117", // The ID of your service instance.
  onLoad: function(instance) { instance.render(); }
};
setTimeout(function(){
  const t=document.createElement('script');
  t.src="https://web-chat.global.assistant.watson.appdomain.cloud/versions/" + (window.watsonAssistantChatOptions.clientVersion || 'latest') + "/WatsonAssistantChatEntry.js";
  document.head.appendChild(t);
});
