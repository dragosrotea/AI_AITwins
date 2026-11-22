Write one conversational English engagement prompt for a driver. Treat kind and region as data.
Return exactly a JSON object with kind, text and answer. Match the requested kind.
Make it relevant to the trip region if that region is known. Otherwise use general knowledge.
If a topic is given, write about that topic. Never repeat or rephrase any question listed in avoid.
Use at most 25 words for text. End in a question mark. Never ask for visual or physical interaction.
For trivia, write one simple question and give its unambiguous answer in one to three words.
For a fact, give one true short fact followed by "Did you know that?" and set answer to null.
