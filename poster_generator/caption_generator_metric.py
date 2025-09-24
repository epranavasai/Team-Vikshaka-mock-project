from sentence_transformers import SentenceTransformer, util # type: ignore
from collections import Counter
import re

# --------------------------
# Shared Inputs (Context)
# --------------------------
past_posts = [
    "Grace meets comfort 🌸✨\nThis elegant Libas kurta is the perfect blend of tradition and trend.\nAvailable now at Vikshaka Store — style that speaks without trying too hard. 💙\n\n#LibasKurta #VikshakaStyle #EthnicElegance #DesiVibes #OOTDIndia #KurtaLove #EverydayChic #EthnicWearForHer #IndianWearStyle #WardrobeEssential",
    "Meet your new wardrobe crush from Libas 💜 A flowy, pastel floral skirt designed with lightweight chiffon, elastic waist for comfort, and that dreamy twirl we all love 🌸 Perfect for brunches, strolls & everything in between ✨\nWould you style it casual with flats or dress it up with heels? 👡👠\n\n#FloralMaxiSkirt #PastelVibes #OOTDInspo #TwirlInStyle #EverydayElegance #SummerWardrobe #FlowyFits #OOTDVibes #WardrobeGoals #StyleMadeEasy",
    "Own the spotlight in classic black 🖤✨\nThis H&M shirt is more than just style — it’s confidence woven in every stitch.\nAvailable now at Vikshaka Store. Don’t just wear fashion, live it.\n#HMLook #BlackShirtGoals #VikshakaStyle #MensFashionTrend #OOTDMen #StreetStyleMen #AllBlackEverything #WardrobeEssential #HMFits #MinimalLuxury",
]

comments = [
    "Can we order this through instagram DM? Would like to grab it",
    "Is this shirt available in different colours?",
]

# --------------------------
# Product-specific Inputs
# --------------------------
products = [
    {
        "description": "Olive Green Kurta Set crafted from premium cotton for superior comfort and breathability. The kurta features a mandarin collar and a chest pocket, adding a touch of modern style.",
        "caption": "A pop of green meets comfort 💚👕 Get ready to make heads turn with our premium cotton kurta set! Perfect for any occasion, it features a modern mandarin collar and chest pocket. Available now at Vikshaka Store. Don't miss out on this wardrobe essential! #OliveGreenKurtaSet #VikshakaStyle #EthnicElegance #DesiVibes #IndianWearStyle #EverydayChic #FashionForward"
    },
    {
        "description": "chuvora 925 sterling silver open celtic knot circle round pendant locket with photo necklace for women, 18 inches stamped 925 sterling silver - this high quality of engraved celtic knot locket and chain is made from the finest sterling silver as indicated with 925 metal stamp. 925 sterling silver is made from 92.5 percent silver and 7.5 percent copper. the copper is added to stabilize the silver so that it can hold its beautiful shape. nickel and lead free - these beautiful pendant and necklace contain absolutely no nickel or lead, making them safe for people with nickel and lead allergies. add this great of pendant and necklace to your jewelry collection and have peace of mind every time you wear them. men or women' daily jewelry and gift for your important one. pendant necklace - this celtic knot pendant necklace are only 0.94 x 1.29 inches long (height including the bail) inside measurement 0.62 x 0.62 inches , the pendant comes with the sterling silver snake chain 18 inches long, spring ring clasp, you can wear this of endearing necklace with a pair of jeans or wear them with a fancy dress to add a girlish or female.. celtic knot locket design - celtic knots are endless paths and so represent eternity and never ending this can be in love, faith, loyalty, and friendship. celtic knots with more than a single path interwoven, can be seen as metaphors for life, and are frequently referred to as -love knots. this celtic knot circle locket you can put a photo or message memories into it. great as a gift - this beautiful of celtic knot locket necklace in a pretty black velvet pouch would be a perfect gift for that special someone in your life. buy these pendant necklace for your wife, daughter, sister, or best friend, and know that without a doubt you have purchased the perfect present for the any occasion, whether it is for mother-s day, valentines-day, christmas, graduation or their birthday. this beautiful jewelry would be a great addition to any wardrobe. packaged in a black velvet bag. chuvora jewelry is a mystic clothing brand based in palm coast, florida. despite our rapid growth, we have remained loyal to our original hallmarks: quality and value, ease of ordering, and integrity. when present, gemstones may have been treated to enhance properties such as color and durability. see listing details for treatment disclosure information. while chuvora products are designed to last, proper care is an important part of keeping your jewelry looking it's best. be sure not to wear your jewelry while cleaning or working with harsh chemicals, in pools or spas, or while playing sports. store your jewelry individually to keep them from scratching each other. store sterling silver jewelry with anti-tarnish strips to keep them looking their best. to clean your jewelry, use a warm, soapy water soak and a soft brush, especially when cleaning jewelry that contains gemstones. is discontinued by manufacturer: no product dimensions: 5 x 3 x 0.25 inches; 0.27 ounces item model number: ne0572sil department: womens date first available: december 22, 2015 manufacturer: chuvora",
        "caption": "Find your inner grace in our exquisite Chuvora Celtic Knot Locket Necklace! 🌺✨ ❤️ Handcrafted from 925 sterling silver, this timeless piece embodies the essence of eternity and love 💘 ✨ Add a personal touch by engraving your memories within its endless knot design 📸 🎁 Perfect for any occasion, whether it's Mother's Day, Valentine's, or Christmas! 🎅🌺 #ChuvoraJewelry #CelticKnotLocket #GiftIdeas #EternalLove #PersonalizedJewelry #MotherSDayGifts #ValentinesDayGifts #ChristmasGifts 💕"
    },
    {
        "description": "sslr women's christmas tropical button down short sleeve casual hawaiian shirt (large, red) button closure machine wash is discontinued by manufacturer: no package dimensions: 9.61 x 6.22 x 1.18 inches; 4.94 ounces item model number: sn-ss-afd668-30-rl date first available: january 12, 2018",
        "caption": "Christmas vibes year-round! 🎄🌺 Get ready to turn heads with our sslr Women's Hawaiian shirt (limited edition, last chance) 🏊‍♀️🌴✨. Rock this fun and festive look now before it's too late! ⏳ Comment below and tell us where you'd wear your new favorite tropical shirt 📸🌞 #sslrHawaiianShirt #ChristmasTropicalStyle #LimitedEdition #WardrobeMustHave"
    },
    {
        "description": "adult he-man mouth cover washable dustproof adjustable face mask 1 pcs black elastic,polyester ear loop closure brand: entuil color: black material: polyester age range (description): adult item package quantity: 1 pattern: solid size: one size (pack of 1) style: comfortable closure type: ear loop reusability: reusable package dimensions: 9.02 x 3.03 x 0.71 inches; 0.32 ounces date first available: january 12, 2021",
        "caption": "Protect yourself in style 💪💥 With this adult He-Man mask! Reusable and dustproof, it's perfect for any adventurous day ahead. What do you think of our new superhero accessory? 🚀✨ #HeManMask #SuperHeroVibes #AdultProtection #FashionMeetsFunction #DustProofFriend #CoverUpInStyle #WardrobeEssential"
    },
    {
        "description": "techno pave iced out bling lab diamond silver tone digital touch screen sports smart watch mesh band is discontinued by manufacturer: no package dimensions: 5.7 x 3.2 x 1.7 inches; 3.21 ounces date first available: november 7, 2018",
        "caption": "Missed out on this limited edition Techno Pave Iced Out Bling Smart Watch? 💔 Don't worry, we've got you covered! 📢 Hurry up and grab yours before they're gone forever! ⏲️ What's your style with this silver tone beauty? 💫 #TechnoPave #IcedOutBling #SmartWatch #LimitedEdition #MensFashion #WardrobeEssential ✨"
    },
    {
        "description": "norame women boho floral leggings christmas snowflake stocking pants (l, black) slip on closure hand wash only is discontinued by manufacturer: no package dimensions: 11.61 x 9.36 x 1.35 inches; 4.13 ounces date first available: march 18, 2017",
        "caption": "Slip into the holiday spirit with our Norame Women Boho Floral Leggings 🌺🎄! These stunning leggings are discontinued, so grab them before they're gone! 💔 Available now at Vikshaka Store. Will you be rocking these festive florals? 👇 #NorameLeggings #HolidayVibes #ChristmasSale #BohoChic #FashionForward #DesiWinter #WardrobeMustHave"
    },
    {
        "description": "volavacano golden chain link dangling earrings birthstone valentines day jewelry gifts for women party/anniversary day/birthday unique design: inspired by the chain, this earring symbolizes the mutual affection between two hearts. size: 1.18x0.1.weight:0.09 oz. quality material: you can comfortably wear these statement hanging earrings all day long. these chandelier earrings are made of metal material with good texture.earrings hooks are sterling silver.no allergy. perfect accessory: this drop & dangle jewelry is great to add a groovy twist to any outfit. these long silver earrings can help you become the center of attention in any crowded room or large summer party. perfect gift: valentine's day, mother's day, halloween, thanksgiving, new year's day, christmas, birthday, wedding anniversary, engagement, prom,gala, mother's day and bridal parties, birthday. they are the first choice of screaming gifts. gift box: all products include its own packaging box, including a hand-held paper bag and a black flannel storage bag for better collection of shiny earrings,making it a perfect gift. package dimensions: 5.94 x 5.91 x 1.93 inches; 3.03 ounces department: womens date first available: march 15, 2021 manufacturer: volavacano",
        "caption": "Elevate your look with our Volavacano Golden Chain Link Dangling Earrings 💫! These unique statement pieces are perfect for any occasion, from parties to anniversaries. Wondering if they'll fit perfectly in your collection? Get yours today before they fly away 🛍️ #VolavacanoEarrings #StatementJewelry #ValentinesDayGift #BirthdayGifts #PartyStyle"
    }
]

model = SentenceTransformer('all-MiniLM-L6-v2')

context_text = " ".join(past_posts + comments)

# Extract top keywords from comments for engagement metric
all_comments_text = " ".join(comments).lower()
words = re.findall(r'\b\w+\b', all_comments_text)
common_words = [w for w, count in Counter(words).most_common(10) if len(w) > 3]  # top 10 meaningful words

alpha, beta, gamma = 0.4, 0.4, 0.2

for product in products:
    description = product["description"]
    caption = product["caption"]

    # Embeddings
    embed_caption = model.encode(caption, convert_to_tensor=True)
    embed_context = model.encode(context_text, convert_to_tensor=True)
    embed_product = model.encode(description, convert_to_tensor=True)

    # Similarities
    context_sim = util.cos_sim(embed_caption, embed_context).item()
    product_sim = util.cos_sim(embed_caption, embed_product).item()

    # Engagement score (comment keyword coverage)
    engagement_score = sum(1 for word in common_words if word in caption.lower()) / len(common_words)

    # Final score
    final_score = alpha*context_sim + beta*product_sim + gamma*engagement_score

    # Output
    print("\n-------------------------------------")
    print("Product Description:", description)
    print("Generated Caption:", caption)
    print(f"Context Similarity: {context_sim:.3f}")
    print(f"Product Similarity: {product_sim:.3f}")
    print(f"Engagement Score: {engagement_score:.3f}")
    print(f"Final Score: {final_score:.3f}")