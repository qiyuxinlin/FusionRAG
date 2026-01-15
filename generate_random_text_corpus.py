#!/usr/bin/env python3
"""
生成无关文本语料库
用于 RANDOM_TEXT 召回模式：BGE召回获取长度，但融合时使用无关文本的KV cache
"""

import json
import os

# 预定义的无关文本库（多种主题，不同长度）
RANDOM_TEXT_CORPUS = [
    # 科学主题
    """The theory of relativity, developed by Albert Einstein, fundamentally changed our understanding of space, time, and gravity.
    Special relativity, published in 1905, introduced the concept that the laws of physics are the same for all non-accelerating observers,
    and that the speed of light in a vacuum is constant regardless of the motion of the light source or observer. This led to the famous
    equation E=mc², demonstrating the equivalence of energy and mass. General relativity, completed in 1915, extended these ideas to
    include acceleration and gravity, describing gravity not as a force but as a curvature of spacetime caused by mass and energy.
    These revolutionary ideas have been confirmed by numerous experiments and observations, including the bending of light around
    massive objects, the precession of Mercury's orbit, and the detection of gravitational waves.""",

    # 历史主题
    """The Renaissance was a period of great cultural change and achievement in Europe that spanned from the 14th to the 17th century.
    It marked the transition from the Middle Ages to modernity and was characterized by an effort to revive and surpass the ideas and
    achievements of classical antiquity. The Renaissance began in Italy and spread throughout Europe, bringing a cultural movement that
    profoundly affected European intellectual life. Artists like Leonardo da Vinci and Michelangelo created masterpieces that are still
    celebrated today. The period also saw major advances in science, with figures like Galileo Galilei challenging traditional views of
    the universe.""",

    # 文学主题
    """Shakespeare's plays have captivated audiences for over four centuries. His works, including tragedies like Hamlet and Macbeth,
    comedies such as A Midsummer Night's Dream, and histories like Henry V, demonstrate an unparalleled understanding of human nature.
    Shakespeare's use of language was revolutionary, introducing countless words and phrases into the English language that we still
    use today. His characters are complex and multi-dimensional, grappling with timeless themes of love, power, jealousy, and mortality.
    The Globe Theatre, where many of his plays were performed, has been reconstructed in London, allowing modern audiences to experience
    his works in a setting similar to the original.""",

    # 技术主题
    """Artificial intelligence has evolved rapidly over the past few decades. Machine learning, a subset of AI, enables computers to
    learn from data without being explicitly programmed. Deep learning, which uses neural networks with multiple layers, has achieved
    remarkable success in tasks such as image recognition, natural language processing, and game playing. Convolutional neural networks
    excel at processing visual information, while recurrent neural networks are particularly effective for sequential data like text and
    speech. The transformer architecture, introduced in 2017, has revolutionized natural language processing, leading to powerful models
    like GPT and BERT. These advances are transforming industries from healthcare to transportation, though they also raise important
    questions about ethics, bias, and the future of work.""",

    # 自然主题
    """The Amazon rainforest, often called the lungs of the Earth, is the world's largest tropical rainforest. Covering approximately
    5.5 million square kilometers, it spans across nine countries in South America, with the majority located in Brazil. The rainforest
    is home to an estimated 390 billion individual trees representing about 16,000 species. This incredible biodiversity includes jaguars,
    sloths, river dolphins, and countless species of birds, insects, and plants, many of which are found nowhere else on Earth. The Amazon
    plays a crucial role in regulating the global climate by absorbing vast amounts of carbon dioxide and producing oxygen. However, it
    faces serious threats from deforestation, primarily driven by agriculture, logging, and mining activities.""",

    # 音乐主题
    """Classical music encompasses a broad period from roughly the 11th century to the present day. The Common Practice Period, spanning
    from about 1650 to 1900, includes the Baroque, Classical, and Romantic eras. Composers like Bach, Mozart, and Beethoven created works
    that defined their respective periods. Bach's intricate fugues demonstrated the complexity of Baroque counterpoint. Mozart's operas
    and symphonies exemplified the clarity and balance of the Classical period. Beethoven bridged the Classical and Romantic eras,
    expanding the emotional range and structural scale of musical works. The 20th century brought modernism and experimentation, with
    composers like Stravinsky and Schoenberg challenging traditional tonality and form.""",

    # 体育主题
    """The Olympic Games represent the pinnacle of athletic competition. Originating in ancient Greece around 776 BCE, the modern Olympics
    were revived in 1896 by Pierre de Coubertin. The Summer and Winter Games now alternate every two years, featuring thousands of athletes
    from around the world competing in dozens of sports. Olympic ideals emphasize not just victory but participation, fair play, and
    international friendship. Notable Olympic moments include Jesse Owens' four gold medals in 1936 Berlin, Nadia Comăneci's perfect 10 in
    gymnastics in 1976, and Usain Bolt's unprecedented sprint dominance in 2008, 2012, and 2016. The Games have also been a platform for
    political statements and social change, from the Black Power salute in 1968 to increased recognition of women's sports.""",

    # 建筑主题
    """Gothic architecture emerged in France during the 12th century and spread throughout Europe until the 16th century. Characterized by
    pointed arches, ribbed vaults, and flying buttresses, Gothic buildings could reach unprecedented heights while maintaining structural
    integrity. Large stained glass windows filled these structures with colored light, creating an ethereal atmosphere meant to inspire
    religious devotion. Notre-Dame de Paris, Chartres Cathedral, and Westminster Abbey are among the most famous examples of Gothic
    architecture. The style represented a significant departure from the earlier Romanesque architecture, with its heavy walls and small
    windows. Gothic cathedrals were not just places of worship but also served as community centers and symbols of civic pride.""",

    # 哲学主题
    """Existentialism is a philosophical movement that emphasizes individual existence, freedom, and choice. Key existentialist thinkers
    include Søren Kierkegaard, Friedrich Nietzsche, Jean-Paul Sartre, and Albert Camus. Existentialists argue that humans define their
    own meaning in life and try to make rational decisions despite existing in an irrational universe. Sartre famously declared that
    "existence precedes essence," meaning that we are not born with a predetermined purpose but must create our own. The philosophy
    emphasizes concepts like authenticity, anxiety, and freedom. Camus explored the absurdity of human existence, arguing that we must
    imagine Sisyphus happy despite his eternal punishment, finding meaning in the struggle itself.""",

    # 经济主题
    """The Industrial Revolution, beginning in Britain in the late 18th century, transformed economies from agrarian to industrial.
    The invention of the steam engine by James Watt revolutionized transportation and manufacturing. Factories replaced cottage industries,
    leading to mass production and urbanization as people moved from rural areas to cities seeking work. This period saw the rise of
    capitalism and new economic theories. Adam Smith's "The Wealth of Nations" laid the foundation for classical economics, arguing for
    free markets and the division of labor. However, the Industrial Revolution also brought challenges, including poor working conditions,
    child labor, and environmental pollution, which eventually led to labor movements and regulatory reforms.""",

    # 地理主题
    """The Himalayas form the highest mountain range in the world, stretching across five countries: Bhutan, India, Nepal, China, and
    Pakistan. Formed by the collision of the Indian and Eurasian tectonic plates about 50 million years ago, the range continues to rise
    by approximately 5mm per year. Mount Everest, known as Sagarmatha in Nepali and Chomolungma in Tibetan, stands at 8,848.86 meters,
    making it the Earth's highest peak above sea level. The Himalayas are the source of major Asian rivers including the Ganges, Indus,
    and Brahmaputra, which provide water to billions of people. The region's unique ecosystems range from tropical forests at lower
    elevations to alpine meadows and glaciers at higher altitudes.""",

    # 心理学主题
    """Cognitive psychology studies mental processes such as perception, memory, problem-solving, and decision-making. Unlike behaviorism,
    which focuses only on observable behavior, cognitive psychology examines the internal mental states that influence how we think and
    act. The cognitive revolution of the 1950s and 1960s, partly inspired by the development of computers, led psychologists to view the
    mind as an information processor. Key concepts include working memory, which temporarily holds and manipulates information; long-term
    memory, where information is stored more permanently; and attention, which determines what information we focus on. Cognitive biases,
    such as confirmation bias and availability heuristic, demonstrate systematic errors in human thinking.""",

    # 天文学主题
    """Black holes are among the most mysterious objects in the universe. Formed when massive stars collapse under their own gravity,
    black holes have gravitational fields so strong that nothing, not even light, can escape once it crosses the event horizon. Einstein's
    theory of general relativity predicted their existence, though the term "black hole" wasn't coined until 1967 by physicist John Wheeler.
    In 2019, the Event Horizon Telescope collaboration captured the first direct image of a black hole's event horizon, located in the
    galaxy M87. Supermassive black holes, millions to billions of times the mass of our Sun, are found at the centers of most galaxies,
    including our own Milky Way. The study of black holes continues to reveal insights about the fundamental nature of space, time,
    and gravity.""",

    # 生物学主题
    """DNA, or deoxyribonucleic acid, carries the genetic instructions for all known living organisms. The discovery of DNA's double helix
    structure by James Watson and Francis Crick in 1953, building on Rosalind Franklin's X-ray crystallography work, revolutionized biology.
    DNA consists of four nucleotide bases—adenine, thymine, guanine, and cytosine—whose sequences encode genetic information. During cell
    division, DNA replicates, ensuring that genetic information passes to daughter cells. The Human Genome Project, completed in 2003,
    mapped all human genes, opening new possibilities for understanding diseases and developing personalized medicine. Modern gene editing
    technologies like CRISPR-Cas9 now allow precise modifications to DNA, raising both exciting possibilities and ethical questions.""",

    # 气候主题
    """Climate change refers to long-term shifts in global temperatures and weather patterns. While climate has changed throughout Earth's
    history, current warming is occurring at an unprecedented rate, primarily due to human activities that release greenhouse gases like
    carbon dioxide and methane. The burning of fossil fuels, deforestation, and industrial agriculture are major contributors. Effects
    include rising sea levels, more frequent extreme weather events, shifts in wildlife populations and habitats, and threats to food
    security. The Paris Agreement, adopted in 2015, aims to limit global temperature increase to well below 2 degrees Celsius above
    pre-industrial levels. Addressing climate change requires a transition to renewable energy, improved energy efficiency, and changes
    in land use and consumption patterns.""",

    # 简短文本 (用于匹配短文档)
    """Photography is the art of capturing light. Since its invention in the 19th century, it has evolved from lengthy exposure times to
    instant digital images. Modern cameras use sensors to record light, converting it to digital data that can be easily stored and shared.""",

    """Coffee is one of the world's most popular beverages. Originating in Ethiopia, coffee cultivation spread to Arabia, Turkey, and
    eventually worldwide. The caffeine in coffee provides a stimulating effect that many people rely on to start their day.""",

    """The periodic table organizes all known chemical elements based on their atomic structure. Dmitri Mendeleev created the first version
    in 1869, leaving gaps for undiscovered elements whose properties he successfully predicted.""",

    """Voting systems vary widely across democracies. Some use first-past-the-post systems, while others employ proportional representation.
    Each system has advantages and disadvantages in terms of fairness and stability.""",

    """Coral reefs are diverse underwater ecosystems built by colonies of tiny animals called coral polyps. They support about 25% of all
    marine species despite covering less than 1% of the ocean floor.""",
]


def generate_random_text_library(output_path="./data/random_text_library.json"):
    """
    生成并保存无关文本库
    """

    # 创建文本库结构
    library = {
        "version": "1.0",
        "description": "Random text corpus for RANDOM_TEXT recall mode",
        "texts": []
    }

    for idx, text in enumerate(RANDOM_TEXT_CORPUS):
        library["texts"].append({
            "id": idx,
            "content": text.strip(),
            "topic": get_topic_name(idx)
        })

    # 保存到文件
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(library, f, indent=2, ensure_ascii=False)

    print(f"✓ Random text library generated: {output_path}")
    print(f"  Total texts: {len(library['texts'])}")
    print(f"  Topics: Science, History, Literature, Technology, Nature, Music, Sports, Architecture, Philosophy, Economics, Geography, Psychology, Astronomy, Biology, Climate, + Short texts")

    return library


def get_topic_name(idx):
    """返回主题名称"""
    topics = [
        "Science", "History", "Literature", "Technology", "Nature",
        "Music", "Sports", "Architecture", "Philosophy", "Economics",
        "Geography", "Psychology", "Astronomy", "Biology", "Climate",
        "Photography", "Coffee", "Chemistry", "Politics", "Marine Biology"
    ]
    return topics[idx] if idx < len(topics) else f"Topic_{idx}"


if __name__ == "__main__":
    library = generate_random_text_library()

    # 打印统计信息
    print("\nText length statistics:")
    for item in library["texts"]:
        length = len(item["content"])
        print(f"  {item['id']:2d}. {item['topic']:15s}: {length:5d} chars")
